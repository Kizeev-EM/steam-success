"""Тесты признаков и схемы валидации (ПР 3)."""
import numpy as np
import pandas as pd

from steam_success.config import load_params
from steam_success.features.build import BASE, FULL, CardFeaturizer
from steam_success.features.history import history_counts
from steam_success.models.cv import time_folds, weekly_top_k

PARAMS = load_params()


def test_publisher_history_respects_12_month_gap():
    """Ручной пример из ПР 1: пять игр одного издателя с известными датами."""
    catalog = pd.DataFrame({
        "publisher": ["P"] * 5,
        "release_date": pd.to_datetime(["2014-01-10", "2014-06-01", "2015-01-10",
                                        "2015-03-01", "2015-12-01"]),
        "y": [1, 0, 0, 1, 1],
    })
    games = catalog[["publisher", "release_date"]]
    h = history_counts(games, catalog, "publisher", gap_months=12)
    # 2015-01-10: ровно 12 месяцев после 2014-01-10 -> одна прошлая игра, успешная.
    # 2015-03-01: та же одна игра (2014-06-01 ещё не созрела).
    # 2015-12-01: созрели 2014-01-10 и 2014-06-01; игры 2015 года не видны.
    assert h["n_prev"].tolist() == [0, 0, 1, 1, 2]
    assert h["n_hit"].tolist() == [0, 0, 1, 1, 1]


def test_unknown_publisher_has_no_history():
    catalog = pd.DataFrame({"publisher": [np.nan, np.nan],
                            "release_date": pd.to_datetime(["2010-01-01", "2011-01-01"]),
                            "y": [1, 1]})
    games = pd.DataFrame({"publisher": [np.nan], "release_date": pd.to_datetime(["2016-01-01"])})
    assert history_counts(games, catalog, "publisher", 12)["n_prev"].tolist() == [0]


def _cards(n_action, n_rare):
    genres = ["Action"] * n_action + ["Rare"] * n_rare
    n = len(genres)
    return pd.DataFrame({
        "genres": genres, "categories": ["Single-player"] * n, "platforms": ["windows"] * n,
        "english": 1, "required_age": 0, "price": 9.99, "achievements": 150,
        "release_date": pd.to_datetime(["2016-12-15"] * n),
        "pub_n_prev": 2, "pub_n_hit": 0, "dev_n_prev": 0, "dev_n_hit": 0,
    })


def test_featurizer_learns_frequent_values_only_from_fit_data():
    fz = CardFeaturizer(FULL, min_count=50).fit(_cards(60, 10))
    assert fz.genres_ == ["Action"]
    X = fz.transform(_cards(1, 1))
    assert "genre_rare" not in X.columns
    assert X["genre_other"].tolist() == [0, 1]
    assert X["ach_capped"].iloc[0] == 100 and X["ach_spam"].iloc[0] == 1
    assert X["pub_only_misses"].iloc[0] == 1
    assert np.isclose(X["month_sin"].iloc[0], np.sin(2 * np.pi * 12 / 12))


def test_base_set_has_no_new_features():
    X = CardFeaturizer(BASE, min_count=50).fit(_cards(60, 0)).transform(_cards(2, 0))
    assert {"price", "achievements", "month"} <= set(X.columns)
    assert not {"pub_had_hit", "price_log", "month_sin"} & set(X.columns)


def test_time_folds_keep_gap_and_order():
    dates = pd.Series(pd.date_range("2010-01-01", "2017-04-30", freq="D"))
    folds = time_folds(dates, PARAMS["cv"])
    assert len(folds) == PARAMS["cv"]["n_folds"]
    for f in folds:
        gap = dates.iloc[f["val"]].min() - dates.iloc[f["train"]].max()
        assert gap.days >= 365          # зазор не меньше года
    assert folds[-1]["val_end"] == "2017-04-30"


def test_weekly_top_k():
    dates = pd.Series(pd.to_datetime(["2016-01-04"] * 4 + ["2016-01-11"] * 2))
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.1, 0.0, 0.5, 0.4])
    p, r = weekly_top_k(y, s, dates, k=2)
    assert p == (0.5 + 0.5) / 2 and r == 2 / 3
