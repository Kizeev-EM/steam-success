"""Очистка и признаки: один модуль для обучения и для новой порции (ПР 3, п. 4).

Обработка делится на две части.

1. prepare() — шаги без обучаемых параметров: отбор модельной выборки,
   очистка по находкам разведочного анализа, удаление повторов по каталогу
   и история издателя и разработчика с зазором. История считается
   point-in-time по каталогу игр с созревшей меткой, поэтому одинакова в
   обучении, в кросс-валидации и в расчёте по порции.
2. CardFeaturizer — шаг scikit-learn с режимами fit и transform. В fit он
   запоминает частые жанры и категории (не реже min_count раз в обучающей
   выборке), в transform строит таблицу числовых признаков. Внутри
   кросс-валидации он обучается заново на каждом обучающем блоке.

Командой `python -m steam_success.features.build` любой файл — исходная
часть или порция — превращается в таблицу признаков data/processed/*.parquet.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from steam_success.config import INCOMING, INTERIM, PROCESSED, REPORTS, load_params
from steam_success.data.io import make_target, read_games, split_list
from steam_success.data.validate import check_cards, model_mask
from steam_success.features.history import history_counts

BASE = "base"   # признаки карточки «как есть»
FULL = "full"   # карточка + девять новых признаков из плана ПР 2 (Таблица 3.2)
NO_HISTORY = "full_no_history"   # полный набор без истории издателя и разработчика
HISTORY = ["pub_had_hit", "pub_only_misses", "pub_n_prev_log", "dev_had_hit"]
NEW_FEATURES = ["pub_had_hit", "pub_only_misses", "pub_n_prev_log", "dev_had_hit",
                "early_access", "price_log", "ach_capped", "ach_spam",
                "month_sin", "month_cos", "online_play"]


# ---------------------------------------------------------------- prepare

def clean(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Очистка карточек, прошедших проверку К1–К10 (Таблица 3.1 ПР 2)."""
    f = params["features"]
    out = df[model_mask(df, params)].copy()          # платные игры без программ
    drop = set(f["drop_categories"])                 # отметка о карточках Steam
    out["categories"] = split_list(out["categories"]).map(
        lambda v: ";".join(c for c in v if c not in drop))
    return out


def catalog_from(labeled: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Каталог для истории: все игры с меткой, включая бесплатные и программы."""
    cat = labeled[["appid", "publisher", "developer", "release_date"]].copy()
    cat["y"] = make_target(labeled["owners"], params["target"]["threshold"])
    for col in ["publisher", "developer"]:
        cat[col] = cat[col].replace("unknown", np.nan)
    return cat


def add_history(games: pd.DataFrame, catalog: pd.DataFrame, params: dict) -> pd.DataFrame:
    gap = params["features"]["history_gap_months"]
    out = games.copy()
    for key, pre in [("publisher", "pub"), ("developer", "dev")]:
        g = out[[key, "release_date"]].assign(**{key: out[key].replace("unknown", np.nan)})
        h = history_counts(g, catalog, key, gap)
        out[f"{pre}_n_prev"], out[f"{pre}_n_hit"] = h["n_prev"], h["n_hit"]
    return out


def drop_seen(games: pd.DataFrame, seen: pd.DataFrame | None) -> tuple[pd.DataFrame, int]:
    """К10 по всему каталогу: пара «название — разработчик» уже встречалась раньше."""
    if seen is None or seen.empty:
        return games, 0
    keys = set(zip(seen["name"], seen["developer"]))
    dup = pd.Series([k in keys for k in zip(games["name"], games["developer"])], index=games.index)
    return games[~dup], int(dup.sum())


def prepare(cards: pd.DataFrame, catalog: pd.DataFrame, params: dict,
            seen: pd.DataFrame | None = None, is_batch: bool = True):
    """Проверка, очистка и история. Возвращает (таблица, отчёт)."""
    checked, rep = check_cards(cards, params, is_batch=is_batch)
    checked, rep["К10_catalog"] = drop_seen(checked, seen)
    out = clean(checked, params)
    out = add_history(out, catalog, params)
    rep["model_rows"] = int(len(out))
    return out.sort_values(["release_date", "appid"], kind="mergesort").reset_index(drop=True), rep


# ---------------------------------------------------------- CardFeaturizer

class CardFeaturizer(BaseEstimator, TransformerMixin):
    """Числовые признаки карточки; обучаемые параметры — списки частых значений."""

    def __init__(self, feature_set: str = FULL, min_count: int = 50, ach_cap: int = 100,
                 online_categories=("Multi-player", "Co-op", "Online Co-op")):
        self.feature_set = feature_set
        self.min_count = min_count
        self.ach_cap = ach_cap
        self.online_categories = online_categories

    def fit(self, X: pd.DataFrame, y=None):
        def frequent(col):
            counts = split_list(X[col]).explode().value_counts()
            return sorted(counts[counts >= self.min_count].index)
        self.genres_ = frequent("genres")
        self.categories_ = frequent("categories")
        self.feature_names_out_ = list(self.transform(X.head(1)).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        g, c, p = split_list(X["genres"]), split_list(X["categories"]), split_list(X["platforms"])
        f = {
            "english": X["english"].astype(int).values,
            "age_restricted": (X["required_age"].astype(int) > 0).astype(int).values,
            "mac": p.map(lambda v: "mac" in v).astype(int).values,
            "linux": p.map(lambda v: "linux" in v).astype(int).values,
            "n_genres": g.map(len).values,
            "n_categories": c.map(len).values,
        }
        # В полном наборе ранний доступ — отдельный признак early_access.
        genres = [x for x in self.genres_ if not (self.feature_set != BASE and x == "Early Access")]
        for name in genres:
            f["genre_" + _slug(name)] = g.map(lambda v, n=name: n in v).astype(int).values
        f["genre_other"] = g.map(lambda v: sum(x not in self.genres_ for x in v)).values
        for name in self.categories_:
            f["cat_" + _slug(name)] = c.map(lambda v, n=name: n in v).astype(int).values
        month = X["release_date"].dt.month.values
        price = X["price"].astype(float).values
        ach = X["achievements"].astype(float).values
        n_prev, n_hit = X["pub_n_prev"].values, X["pub_n_hit"].values
        if self.feature_set == BASE:
            f.update(price=price, achievements=ach, month=month)
        else:
            f.update(
                price_log=np.log1p(price),
                ach_capped=np.minimum(ach, self.ach_cap),
                ach_spam=(ach > self.ach_cap).astype(int),
                month_sin=np.sin(2 * np.pi * month / 12),
                month_cos=np.cos(2 * np.pi * month / 12),
                early_access=g.map(lambda v: "Early Access" in v).astype(int).values,
                online_play=c.map(lambda v: bool(set(self.online_categories) & set(v))).astype(int).values,
                pub_had_hit=(n_hit > 0).astype(int),
                pub_only_misses=((n_prev > 0) & (n_hit == 0)).astype(int),
                pub_n_prev_log=np.log1p(n_prev),
                dev_had_hit=(X["dev_n_hit"].values > 0).astype(int),
            )
        out = pd.DataFrame(f, index=X.index).astype(float)
        if self.feature_set == NO_HISTORY:
            out = out.drop(columns=HISTORY)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s.lower()).strip("_")


# ------------------------------------------------------------ загрузка

def load_initial(params: dict):
    """Исходная часть: таблица после prepare, метки и каталог для истории."""
    initial = read_games(INTERIM / "initial.csv")
    checked, _ = check_cards(initial, params, is_batch=False)
    catalog = catalog_from(checked, params)
    games, rep = prepare(initial, catalog, params, is_batch=False)
    games = games.merge(catalog[["appid", "y"]], on="appid", how="left")
    return games, catalog, rep


def prepare_games(path: Path | str, params: dict | None = None, previous=()):
    """Файл карточек -> проверенная и очищенная таблица с историей издателя.

    Это вход итоговой модели: models/model_v1.joblib сам строит признаки
    шагом CardFeaturizer. Каталог для истории и проверки повторов — исходная
    часть с метками (для порций тестового года более поздние метки ещё не
    созрели); повторы ищутся также в уже принятых порциях previous.
    """
    params = params or load_params()
    path = Path(path)
    train, catalog, rep = load_initial(params)
    if path.resolve() == (INTERIM / "initial.csv").resolve():
        return train, train, rep
    seen = pd.concat([read_games(p)[["name", "developer"]]
                      for p in [INTERIM / "initial.csv", *previous]]).fillna("unknown")
    games, rep = prepare(read_games(path), catalog, params, seen=seen)
    return games, train, rep


def prepare_file(path: Path | str, params: dict | None = None, previous=()):
    """Одна команда: любой файл карточек -> таблица числовых признаков.

    Список частых жанров и категорий берётся из исходной части.
    """
    params = params or load_params()
    games, train, rep = prepare_games(path, params, previous)
    fz = CardFeaturizer(FULL, params["features"]["min_count"], params["features"]["ach_cap"],
                        tuple(params["features"]["online_categories"]))
    fz.fit(train)
    X = fz.transform(games)
    X.insert(0, "appid", games["appid"].values)
    X.insert(1, "release_date", games["release_date"].values)
    if "y" in games:
        X["y"] = games["y"].values
    return X, fz, rep


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    params = load_params()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    files = [INTERIM / "initial.csv"] + [INCOMING / f"{b['name']}.csv"
                                         for b in params["split"]["batches"]]
    report, fz = {}, None
    for i, path in enumerate(files):
        previous = files[1:i]                      # порции, принятые раньше этой
        X, fz, rep = prepare_file(path, params, previous)
        out = PROCESSED / f"features_{path.stem}.parquet"
        X.to_parquet(out, index=False)
        # Проверка повторяемости: второй запуск на тех же данных.
        X2, _, _ = prepare_file(path, params, previous)
        same = X.equals(X2)
        report[path.stem] = {"rows": int(len(X)), "columns": int(X.shape[1]),
                             "duplicates_in_catalog": rep.get("К10_catalog", 0),
                             "md5": _md5(out), "second_run_identical": bool(same)}
    with open(PROCESSED / "feature_params.json", "w", encoding="utf-8") as f:
        json.dump({"genres": fz.genres_, "categories": fz.categories_,
                   "features": fz.feature_names_out_}, f, ensure_ascii=False, indent=1)
    with open(REPORTS / "features_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    for k, v in report.items():
        print(k, v)


if __name__ == "__main__":
    main()
