"""История издателя и разработчика с зазором (point-in-time).

Для игры, вышедшей в дату t, учитываются только релизы того же издателя
(разработчика), вышедшие не позже t - gap_months: только у них к моменту t
известна метка. Каталог — все игры с созревшей меткой, включая бесплатные
и программы: они тоже говорят об опыте издателя.
"""
import numpy as np
import pandas as pd


def history_counts(games: pd.DataFrame, catalog: pd.DataFrame, key: str,
                   gap_months: int) -> pd.DataFrame:
    """Возвращает для каждой игры число прошлых релизов и число успешных.

    games   — таблица с полями key и release_date;
    catalog — таблица с полями key, release_date и y (метка успеха).
    Пропуск в key означает «издатель неизвестен»: истории нет.
    """
    cat = catalog.dropna(subset=[key]).sort_values("release_date", kind="mergesort")
    index = {k: (g["release_date"].values, np.cumsum(g["y"].values))
             for k, g in cat.groupby(key, sort=False)}
    cutoff = (games["release_date"] - pd.DateOffset(months=gap_months)).values
    n_prev = np.zeros(len(games), dtype=int)
    n_hit = np.zeros(len(games), dtype=int)
    for i, (k, c) in enumerate(zip(games[key].values, cutoff)):
        if pd.isna(k) or k not in index:
            continue
        dates, cum_y = index[k]
        n = np.searchsorted(dates, c, side="right")   # релизы с датой <= t - gap
        if n:
            n_prev[i] = n
            n_hit[i] = cum_y[n - 1]
    return pd.DataFrame({"n_prev": n_prev, "n_hit": n_hit}, index=games.index)
