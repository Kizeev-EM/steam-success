"""Схема валидации и метрики (ПР 3, п. 1).

Расширяющееся окно по дате выхода: модель проверяется на шести месяцах и
обучается только на играх, вышедших не позже чем за gap_months до начала
проверки. Зазор повторяет условие эксплуатации: в момент прогноза известны
метки только игр, вышедших год назад и раньше.
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score


def time_folds(dates: pd.Series, cv: dict) -> list[dict]:
    """Список блоков: индексы обучения и проверки и границы периодов."""
    # Границы считаются от начала месяца: проверка — полуинтервал [v_start, v_stop).
    stop = pd.Timestamp(cv["last_fold_end"]) + pd.Timedelta(days=1)
    step = pd.DateOffset(months=cv["fold_months"])
    folds = []
    for k in range(cv["n_folds"]):
        v_start = stop - step * (cv["n_folds"] - k)
        v_stop = v_start + step
        t_stop = v_start - pd.DateOffset(months=cv["gap_months"])
        train = np.where(dates < t_stop)[0]
        val = np.where((dates >= v_start) & (dates < v_stop))[0]
        folds.append({"fold": k + 1,
                      "train_end": (t_stop - pd.Timedelta(days=1)).date().isoformat(),
                      "val_start": v_start.date().isoformat(),
                      "val_end": (v_stop - pd.Timedelta(days=1)).date().isoformat(),
                      "train": train, "val": val})
    return folds


def weekly_top_k(y: np.ndarray, score: np.ndarray, dates: pd.Series, k: int) -> tuple[float, float]:
    """Недельный список: точность и полнота среди k игр недели с наибольшей оценкой.

    Так работает редакция: каждую неделю выбирает k слотов из новинок недели.
    Точность усредняется по неделям, полнота считается по всем успешным играм периода.
    """
    df = pd.DataFrame({"y": y, "s": score, "w": dates.dt.to_period("W").astype(str).values})
    df["rank"] = df.groupby("w")["s"].rank(ascending=False, method="first")
    top = df[df["rank"] <= k]
    precision = top.groupby("w")["y"].mean().mean()
    recall = top["y"].sum() / max(df["y"].sum(), 1)
    return float(precision), float(recall)


def fold_metrics(y, score, dates, top_k) -> dict:
    p_at_k, r_at_k = weekly_top_k(y, score, dates, top_k)
    base = float(np.mean(y))
    pr_auc = float(average_precision_score(y, score))
    return {"pr_auc": pr_auc, "base_rate": base, "pr_auc_lift": pr_auc / base,
            "roc_auc": float(roc_auc_score(y, score)),
            f"precision_at_{top_k}w": p_at_k, f"recall_at_{top_k}w": r_at_k}


def recency_weights(dates: pd.Series, half_life_years) -> np.ndarray | None:
    """Веса недавних лет: вес игры вдвое меньше каждые half_life_years лет назад."""
    if not half_life_years:
        return None
    age = (dates.max() - dates).dt.days.values / 365.25
    return 0.5 ** (age / half_life_years)


def cross_validate(pipe, X: pd.DataFrame, y: np.ndarray, folds: list[dict], top_k: int,
                   half_life=None, score_fn=None, with_train: bool = False) -> pd.DataFrame:
    """Обучает копию пайплайна на каждом блоке и считает метрики.

    score_fn — для правил без обучения (базовые решения): score_fn(X_val) -> оценка.
    """
    rows = []
    for f in folds:
        Xt, Xv = X.iloc[f["train"]], X.iloc[f["val"]]
        yt, yv = y[f["train"]], y[f["val"]]
        t0 = time.perf_counter()
        if score_fn is None:
            model = clone(pipe)
            w = recency_weights(Xt["release_date"], half_life)
            fit_kw = {"model__sample_weight": w} if w is not None else {}
            model.fit(Xt, yt, **fit_kw)
            fit_time = time.perf_counter() - t0
            score = model.predict_proba(Xv)[:, 1]
        else:
            fit_time, score = 0.0, score_fn(Xv)
        m = fold_metrics(yv, score, Xv["release_date"], top_k)
        m.update(fold=f["fold"], n_train=len(yt), n_val=len(yv), fit_time=fit_time)
        if with_train and score_fn is None:
            m["train_pr_auc"] = float(average_precision_score(yt, model.predict_proba(Xt)[:, 1]))
        rows.append(m)
    return pd.DataFrame(rows)
