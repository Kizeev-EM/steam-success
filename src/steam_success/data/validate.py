"""Стадия validate: проверка требований К1–К13 (Таблица 1.5 ПР 2).

Остановочные нарушения (К1, К2) прекращают обработку порции целиком,
построчные (К3, К4, К5, К8) исключают запись, остальные дают предупреждение
или приводят значение к допустимому. Итог пишется в reports/quality.json,
время работы проверок — в reports/timing.json (DVC за ним не следит).
"""
import json
import time

import numpy as np
import pandas as pd

from steam_success.config import INCOMING, INTERIM, REPORTS, load_params
from steam_success.data.io import make_target, read_games, split_list
from steam_success.data.schema import card_row_checks, card_stop_schema, labels_schema
from steam_success.features.history import history_counts

POST_RELEASE = ["positive_ratings", "negative_ratings", "average_playtime",
                "median_playtime", "steamspy_tags", "owners"]


class BatchRejected(Exception):
    """Остановочное нарушение: порция не обрабатывается (F-03)."""


def is_software(genres: pd.Series, software: list[str]) -> pd.Series:
    soft = set(software)
    return split_list(genres).map(lambda g: bool(soft & set(g)))


def model_mask(df: pd.DataFrame, params: dict) -> pd.Series:
    """Модельная выборка: платные игры без жанров-программ."""
    return (df["price"] > 0) & ~is_software(df["genres"], params["quality"]["software_genres"])


def check_cards(df: pd.DataFrame, params: dict, is_batch: bool = True):
    """Проверяет карточки. Возвращает (очищенная таблица, отчёт).

    Бросает BatchRejected при нарушении К1 или К2.
    """
    q = params["quality"]
    card_fields = params["split"]["card_fields"]
    dump = pd.Timestamp(params["data"]["dump_date"])
    rep = {"rows_in": int(len(df))}

    # К1: все поля карточки на месте; в порции нет полей, известных после релиза.
    missing = [c for c in card_fields if c not in df.columns]
    extra = [c for c in POST_RELEASE if c in df.columns] if is_batch else []
    rep["К1"] = {"missing": missing, "post_release": extra}
    if missing or extra:
        raise BatchRejected(f"К1: нет полей {missing}, лишние поля {extra}")

    # К2: appid целый, уникальный, не пустой.
    try:
        card_stop_schema.validate(df[["appid"]])
    except Exception as e:  # pandera.errors.SchemaError
        raise BatchRejected(f"К2: {str(e).splitlines()[0]}") from e
    rep["К2"] = 0

    # Построчные требования.
    bad = pd.Series(False, index=df.index)
    for code, check in card_row_checks.items():
        ok = check(df, dump).fillna(False).astype(bool)
        rep[code] = int((~ok).sum())
        bad |= ~ok
    rep["К4_review_over_100"] = int((df["price"] > q["price_review"]).sum())
    out = df[~bad].copy()

    # К6: жанры-программы — вне модельной выборки (счётчик).
    rep["К6"] = int(is_software(out["genres"], q["software_genres"]).sum())

    # К7: возраст приводится к ближайшему допустимому значению.
    ages = np.array(q["allowed_ages"])
    age = pd.to_numeric(out["required_age"], errors="coerce").fillna(0)
    fixed = age.map(lambda a: int(ages[np.abs(ages - a).argmin()]))
    rep["К7"] = int((fixed != age).sum())
    out["required_age"] = fixed

    # К9: пропуски разработчика и издателя — unknown; доля выше порога — предупреждение.
    for col in ["developer", "publisher"]:
        n_miss = int(out[col].isna().sum())
        rep[f"К9_{col}"] = n_miss
        rep[f"К9_{col}_warn"] = bool(n_miss > q["max_missing_share"] * len(out))
        out[col] = out[col].fillna("unknown")

    # К10: повтор пары «название — разработчик» внутри файла, остаётся ранняя запись.
    dup = out.sort_values(["release_date", "appid"]).duplicated(["name", "developer"])
    rep["К10"] = int(dup.sum())
    out = out.drop(index=dup[dup].index)

    rep["rows_out"] = int(len(out))
    return out, rep


def check_labels(labels: pd.DataFrame) -> dict:
    """К11: owners из набора диапазонов SteamSpy, иначе файл меток не принимается."""
    labels_schema.validate(labels[["appid", "owners"]])
    return {"К11": 0}


def new_publisher_share(games: pd.DataFrame, catalog: pd.DataFrame, params: dict) -> float:
    """К12: доля игр модельной выборки, у издателя которых нет истории."""
    g = games.assign(publisher=games["publisher"].replace("unknown", np.nan))
    c = catalog.assign(publisher=catalog["publisher"].replace("unknown", np.nan))
    h = history_counts(g, c, "publisher", params["features"]["history_gap_months"])
    return float((h["n_prev"] == 0).mean())


def run() -> dict:
    params = load_params()
    q, sp, tgt = params["quality"], params["split"], params["target"]
    timing, report = {}, {}

    t0 = time.perf_counter()
    initial = read_games(INTERIM / "initial.csv")
    initial_ok, rep = check_cards(initial, params, is_batch=False)
    rep.update(check_labels(initial))
    initial_ok["y"] = make_target(initial_ok["owners"], tgt["threshold"])
    catalog = initial_ok[["publisher", "developer", "release_date", "y"]]
    model = initial_ok[model_mask(initial_ok, params)]
    rep["model_rows"] = int(len(model))
    rep["success_share"] = round(float(model["y"].mean()), 4)
    rep["К12"] = round(new_publisher_share(model, catalog, params), 4)
    rep["К12_warn"] = rep["К12"] > q["max_new_publisher_share"]
    report["initial"] = rep
    timing["initial"] = round(time.perf_counter() - t0, 3)

    all_labeled = initial_ok
    for b in sp["batches"]:
        name, k = b["name"], b["name"].split("_")[1]
        t0 = time.perf_counter()
        cards = read_games(INCOMING / f"{name}.csv")
        try:
            ok, rep = check_cards(cards, params, is_batch=True)
        except BatchRejected as e:
            report[name] = {"rejected": str(e)}
            continue
        model = ok[model_mask(ok, params)]
        rep["model_rows"] = int(len(model))
        rep["К12"] = round(new_publisher_share(model, catalog, params), 4)
        rep["К12_warn"] = rep["К12"] > q["max_new_publisher_share"]
        timing[name] = round(time.perf_counter() - t0, 3)

        # Метки порции: К11 и К13 (проверяются, когда файл меток пришёл).
        labels = pd.read_csv(INCOMING / f"labels_{k}.csv", dtype={"owners": str})
        rep.update(check_labels(labels))
        lab = model.merge(labels[["appid", "owners"]], on="appid")
        lab["y"] = make_target(lab["owners"], tgt["threshold"])
        start = pd.Timestamp(b["start"])
        prev = all_labeled[(all_labeled["release_date"] < start)
                           & (all_labeled["release_date"] >= start - pd.DateOffset(months=12))]
        prev = prev[model_mask(prev, params)]
        share, prev_share = float(lab["y"].mean()), float(prev["y"].mean())
        rep["success_share"] = round(share, 4)
        rep["success_share_prev12m"] = round(prev_share, 4)
        rep["К13_ratio"] = round(prev_share / share, 2)
        rep["К13_warn"] = rep["К13_ratio"] > q["max_success_ratio"]
        report[name] = rep
        ok_lab = ok.merge(labels[["appid", "owners"]], on="appid")
        ok_lab["y"] = make_target(ok_lab["owners"], tgt["threshold"])
        all_labeled = pd.concat([all_labeled, ok_lab], ignore_index=True)

    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / "quality.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    with open(REPORTS / "timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=1)
    return report


def main() -> None:
    report = run()
    for name, rep in report.items():
        print(name, {k: v for k, v in rep.items()
                     if k.startswith(("К4_", "К6", "К9_pub", "К10", "К12", "К13", "success"))})


if __name__ == "__main__":
    main()
