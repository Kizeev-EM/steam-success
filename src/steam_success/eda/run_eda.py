"""Стадия eda: разведочный анализ исходной части (ПР 2, раздел 1).

Все числа отчёта складываются в reports/eda_summary.json, рисунки — в
reports/figures. Анализ ведётся только по исходной части.
"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from steam_success.config import FIGURES, INTERIM, REPORTS, load_params
from steam_success.data.io import make_target, owners_lower, read_games, split_list
from steam_success.data.validate import check_cards, is_software
from steam_success.features.history import history_counts

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def r(x, n=4):
    return round(float(x), n)


def odds_ratio(df: pd.DataFrame, formula: str, term: str) -> dict:
    """Отношение шансов терма в логистической регрессии с 95 % ДИ."""
    m = smf.logit(formula, data=df).fit(disp=0)
    ci = np.exp(m.conf_int().loc[term])
    return {"or": r(np.exp(m.params[term]), 3), "ci": [r(ci[0], 3), r(ci[1], 3)],
            "p": float(m.pvalues[term])}


def build_frame(params: dict):
    initial = read_games(INTERIM / "initial.csv")
    ok, _ = check_cards(initial, params, is_batch=False)
    ok["y"] = make_target(ok["owners"], params["target"]["threshold"])
    ok["year"] = ok["release_date"].dt.year
    # Игры 2012 года и раньше объединены в одну группу: до 2012 года
    # в год выходит меньше 300 платных игр, и оценки по отдельным годам шумные.
    ok["year_g"] = ok["year"].clip(lower=2012)
    q = params["quality"]
    soft = is_software(ok["genres"], q["software_genres"])
    df = ok[(ok["price"] > 0) & ~soft].copy()
    g, c = split_list(df["genres"]), split_list(df["categories"])
    df["n_genres"], df["n_categories"] = g.map(len), c.map(len)
    df["cards"] = c.map(lambda v: "Steam Trading Cards" in v).astype(int)
    df["early_access"] = g.map(lambda v: "Early Access" in v).astype(int)
    df["vr"] = c.map(lambda v: "VR Support" in v).astype(int)
    df["online"] = c.map(lambda v: bool({"Multi-player", "Co-op", "Online Co-op"} & set(v))).astype(int)
    df["workshop"] = c.map(lambda v: "Steam Workshop" in v).astype(int)
    df["has_ach"] = c.map(lambda v: "Steam Achievements" in v).astype(int)
    p = split_list(df["platforms"])
    df["mac"], df["linux"] = p.map(lambda v: "mac" in v).astype(int), p.map(lambda v: "linux" in v).astype(int)
    cat = ok[["publisher", "release_date", "y"]].assign(
        publisher=ok["publisher"].replace("unknown", np.nan))
    h = history_counts(df.assign(publisher=df["publisher"].replace("unknown", np.nan)),
                       cat, "publisher", params["features"]["history_gap_months"])
    df["pub_n_prev"] = h["n_prev"]
    df["pub_had_hit"] = (h["n_hit"] > 0).astype(int)
    df["pub_only_misses"] = ((h["n_prev"] > 0) & (h["n_hit"] == 0)).astype(int)
    df["price_group"] = pd.cut(df["price"], [0, 0.99, 4.99, 9.99, 19.99, 1e9],
                               labels=["до 0,99", "1–4,99", "5–9,99", "10–19,99", "20+"])
    return ok, df, g


def main() -> None:
    warnings.filterwarnings("ignore")
    params = load_params()
    ok, df, genres = build_frame(params)
    FIGURES.mkdir(parents=True, exist_ok=True)
    s = {}

    # --- Состав набора (подраздел 1.3) ---
    s["records"] = len(ok)
    s["model_rows"], s["success"] = len(df), int(df["y"].sum())
    s["success_share"] = r(df["y"].mean())
    s["date_min"], s["date_max"] = str(ok["release_date"].min().date()), str(ok["release_date"].max().date())
    s["share_2014_2017"] = r((ok["year"] >= 2014).mean())
    # Показатели Таблицы 1.2 считаются по модельной выборке.
    pub_games = df[df["publisher"] != "unknown"].groupby("publisher").size()
    s["n_publishers"], s["publishers_one_game"] = int(len(pub_games)), r((pub_games == 1).mean())
    s["self_published"] = r((df["publisher"] == df["developer"]).mean())
    s["median_genres"] = float(df["n_genres"].median())
    s["median_categories"] = float(df["n_categories"].median())
    s["age_restricted"] = r((df["required_age"] > 0).mean())
    s["median_price_paid"] = float(df["price"].median())
    s["no_achievements"] = r((df["achievements"] == 0).mean())
    low = owners_lower(ok.loc[ok["price"] > 0, "owners"])   # все платные игры
    s["owners_0_20k"], s["owners_20_50k"] = r((low == 0).mean()), r((low == 20000).mean())
    free = ok[ok["price"] == 0]
    s["free_games"], s["free_success_share"] = len(free), r(free["y"].mean())
    s["success_by_year"] = {("<=2012" if y == 2012 else str(y)): r(v)
                            for y, v in df.groupby("year_g")["y"].mean().items()}
    early17 = df[df["release_date"] >= "2017-01-01"]
    s["success_2017_jan_apr"] = r(early17["y"].mean())

    # Рисунок 1.1: доля успешных и доля игр с карточками по полугодиям.
    per = df[df["release_date"] >= "2010-01-01"].copy()
    per["half"] = per["release_date"].dt.year.astype(str) + np.where(
        per["release_date"].dt.month <= 6, "-1", "-2")
    agg = per.groupby("half").agg(success=("y", "mean"), cards=("cards", "mean"), n=("y", "size"))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(agg.index, agg["success"] * 100, marker="o", label="Доля успешных игр")
    ax.plot(agg.index, agg["cards"] * 100, marker="s", linestyle="--", label="Доля игр с карточками Steam")
    ax.set_xlabel("Полугодие выхода игры")
    ax.set_ylabel("Доля, %")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_success_by_period.png", dpi=200)
    plt.close(fig)

    # --- Проблемы качества (подраздел 1.4) ---
    def iqr_share(x):
        q1, q3 = x.quantile([0.25, 0.75])
        return r(((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).mean())
    s["price_iqr_outliers"], s["ach_iqr_outliers"] = iqr_share(df["price"]), iqr_share(df["achievements"])
    s["price_skew"], s["ach_skew"] = r(stats.skew(df["price"]), 2), r(stats.skew(df["achievements"]), 1)
    s["ach_over_100"] = int((df["achievements"] > 100).sum())
    s["ach_over_1000"] = int((df["achievements"] > 1000).sum())
    s["ach_max"] = int(df["achievements"].max())
    s["cards_records"] = int(df["cards"].sum())
    s["no_english"] = int((df["english"] == 0).sum())

    # --- Связи с успехом (подраздел 1.5) ---
    succ, fail = df[df["y"] == 1], df[df["y"] == 0]
    s["median_price"] = [float(succ["price"].median()), float(fail["price"].median())]
    s["median_ach"] = [float(succ["achievements"].median()), float(fail["achievements"].median())]
    s["mw_price_p"] = float(stats.mannwhitneyu(succ["price"], fail["price"]).pvalue)
    s["success_by_price_group"] = {str(k): r(v) for k, v in df.groupby("price_group", observed=True)["y"].mean().items()}
    hist = np.select([df["pub_had_hit"] == 1, df["pub_only_misses"] == 1], ["hit", "misses"], "none")
    s["success_by_history"] = {k: r(v) for k, v in df.groupby(hist)["y"].mean().items()}
    s["share_no_history"] = r((hist == "none").mean())
    gl = split_list(df["genres"])
    genre_share = {}
    for gname in ["Action", "Adventure", "Casual", "Indie", "RPG", "Simulation", "Strategy", "Early Access"]:
        m = gl.map(lambda v: gname in v)
        genre_share[gname] = r(df.loc[m, "y"].mean())
    s["success_by_genre"] = genre_share
    s["vr_games"], s["vr_success"] = int(df["vr"].sum()), r(df.loc[df["vr"] == 1, "y"].mean())
    cl = split_list(df["categories"])
    s["success_by_category"] = {c: r(df.loc[cl.map(lambda v, c=c: c in v), "y"].mean())
                                for c in ["Multi-player", "Co-op", "Online Co-op", "Steam Workshop"]}

    # Рисунок 1.2: доля успешных в разрезах.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    pg = df.groupby("price_group", observed=True)["y"].mean() * 100
    axes[0].bar(pg.index.astype(str), pg.values, color="#4C72B0")
    axes[0].set_title("Цена, долл.")
    hs = pd.Series(s["success_by_history"]).reindex(["hit", "misses", "none"]) * 100
    axes[1].bar(["был успех", "только неудачи", "нет истории"], hs.values, color="#55A868")
    axes[1].set_title("История издателя")
    gs = pd.Series(genre_share).sort_values() * 100
    axes[2].barh(gs.index, gs.values, color="#C44E52")
    axes[2].set_title("Жанр")
    for a in axes:
        a.axhline(df["y"].mean() * 100, color="black", linestyle="--", linewidth=1) if a is not axes[2] else \
            a.axvline(df["y"].mean() * 100, color="black", linestyle="--", linewidth=1)
        a.grid(alpha=0.3)
    axes[0].set_ylabel("Доля успешных игр, %")
    axes[2].set_xlabel("Доля успешных игр, %")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_success_by_group.png", dpi=200)
    plt.close(fig)

    # Рисунок 1.3: ранговые корреляции Спирмена.
    cols = {"y": "успех", "year": "год выхода", "price": "цена", "achievements": "достижения",
            "has_ach": "отметка достижений", "n_genres": "число жанров", "n_categories": "число категорий",
            "pub_n_prev": "релизов издателя", "pub_had_hit": "прошлый успех издателя",
            "cards": "карточки Steam", "early_access": "ранний доступ", "online": "мультиплеер",
            "mac": "macOS", "linux": "Linux", "vr": "VR"}
    corr = df[list(cols)].corr(method="spearman")
    s["spearman_with_y"] = {k: r(v, 2) for k, v in corr["y"].items() if k != "y"}
    s["spearman_pairs"] = {"pub_n_prev~pub_had_hit": r(corr.loc["pub_n_prev", "pub_had_hit"], 2),
                           "achievements~has_ach": r(corr.loc["achievements", "has_ach"], 2),
                           "mac~linux": r(corr.loc["mac", "linux"], 2)}
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), list(cols.values()), rotation=60, ha="right")
    ax.set_yticks(range(len(cols)), list(cols.values()))
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, label="Коэффициент Спирмена")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_spearman.png", dpi=200)
    plt.close(fig)

    # --- Гипотезы (подраздел 1.7): логит с поправкой на год выхода ---
    d = df.copy()
    d["cheap"] = (d["price"] <= 0.99).astype(int)
    hyp = {}
    hyp["G1_cards"] = odds_ratio(d, "y ~ cards + C(year_g)", "cards")
    hyp["G1_shares"] = [r(d.loc[d.cards == 1, "y"].mean()), r(d.loc[d.cards == 0, "y"].mean())]
    m = smf.logit("y ~ pub_had_hit + pub_only_misses + C(year_g)", data=d).fit(disp=0)
    ci = np.exp(m.conf_int())
    hyp["G2_hit"] = {"or": r(np.exp(m.params["pub_had_hit"]), 3), "ci": [r(v, 3) for v in ci.loc["pub_had_hit"]]}
    hyp["G2_misses"] = {"or": r(np.exp(m.params["pub_only_misses"]), 3), "ci": [r(v, 3) for v in ci.loc["pub_only_misses"]]}
    # Г3: поправка на состав потока — восемь самых частых жанров, ранний доступ,
    # ценовая группа, многопользовательские режимы, платформы, история издателя
    # и число достижений.
    d = d.assign(log_ach=np.log1p(d["achievements"].clip(upper=100)),
                 pub_log=np.log1p(d["pub_n_prev"]))
    gl = split_list(d["genres"])
    top8 = [x for x in gl.explode().value_counts().index if x != "Early Access"][:8]
    genre_cols = []
    for gname in top8:
        col = "g_" + "".join(ch for ch in gname.lower() if ch.isalnum())
        d[col] = gl.map(lambda v, gname=gname: gname in v).astype(int)
        genre_cols.append(col)
    raw = smf.logit("y ~ C(year_g)", data=d).fit(disp=0)
    adj = smf.logit("y ~ C(year_g) + " + " + ".join(genre_cols)
                    + " + early_access + C(price_group) + online + mac + linux"
                      " + pub_had_hit + pub_only_misses + pub_log + log_ach", data=d).fit(disp=0)
    hyp["G3_year_or_vs_le2012"] = {str(y): [r(np.exp(raw.params[f"C(year_g)[T.{y}]"]), 2),
                                            r(np.exp(adj.params[f"C(year_g)[T.{y}]"]), 2)]
                                   for y in [2013, 2014, 2015, 2016, 2017]}
    hyp["G4_early_access"] = odds_ratio(d, "y ~ early_access + C(year_g)", "early_access")
    hyp["G5_cheap"] = odds_ratio(d, "y ~ cheap + C(year_g)", "cheap")
    hyp["G5_cheap_shares"] = [r(d.loc[d.cheap == 1, "y"].mean()), r(d.loc[d.cheap == 0, "y"].mean())]
    nc = d[d["cards"] == 0]
    hyp["G5_cheap_no_cards"] = odds_ratio(nc, "y ~ cheap + C(year_g)", "cheap")
    hyp["G5_cheap_with_cards_share"] = r(d.loc[(d.cheap == 1) & (d.cards == 1), "y"].mean())
    hyp["G5_cheap_no_cards_share"] = r(d.loc[(d.cheap == 1) & (d.cards == 0), "y"].mean())
    s["hypotheses"] = hyp

    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
    print(json.dumps(s, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
