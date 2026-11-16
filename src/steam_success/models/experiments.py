"""Стадия experiments: базовые решения и модели пяти семейств (ПР 3, пп. 5–7).

Все модели сравниваются на одних и тех же пяти временных блоках исходной
части. Отложенные порции в этой работе не используются.
Результаты: reports/experiments.csv (сводка), reports/experiments_folds.csv
(по блокам), reports/reg_path.csv, reports/tree_depth.csv,
reports/logreg_coefs.csv и рисунки; каждый эксперимент записывается в MLflow.
"""
import json
import warnings

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from steam_success.config import FIGURES, REPORTS, load_params
from steam_success.features.build import BASE, FULL, NO_HISTORY, load_initial
from steam_success.models import tracking, zoo
from steam_success.models.cv import cross_validate, time_folds

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
COMMA = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
METRICS = ["pr_auc", "pr_auc_lift", "roc_auc"]


def num(x) -> str:
    """Число в подписи: десятичная запятая."""
    return f"{x:g}".replace(".", ",")


def summarize(name, family, feature_set, desc, folds_df, top_k) -> dict:
    s = {"experiment": name, "family": family, "feature_set": feature_set, "params": desc}
    for m in METRICS + [f"precision_at_{top_k}w", f"recall_at_{top_k}w"]:
        s[m + "_mean"] = folds_df[m].mean()
    s["pr_auc_std"] = folds_df["pr_auc"].std(ddof=0)
    return s


class Runner:
    def __init__(self, params):
        self.params = params
        self.cv = params["cv"]
        self.games, _, _ = load_initial(params)
        self.y = self.games["y"].values
        self.folds = time_folds(self.games["release_date"], self.cv)
        self.summary, self.fold_rows, self.timing = [], [], {}
        tracking.start(params)

    def run(self, name, family, pipe=None, feature_set=FULL, desc="", score_fn=None,
            with_train=False, record=True):
        df = cross_validate(pipe, self.games, self.y, self.folds, self.cv["top_k"],
                            score_fn=score_fn, with_train=with_train)
        if record:
            s = summarize(name, family, feature_set, desc, df, self.cv["top_k"])
            self.summary.append(s)
            self.fold_rows.append(df.drop(columns="fit_time").assign(experiment=name))
            self.timing[name] = round(float(df["fit_time"].mean()), 3)
            tracking.log_experiment(name, {"family": family, "feature_set": feature_set,
                                           "params": desc}, df, s, stage="experiments")
            print(f"{name:42s} PR-AUC {s['pr_auc_mean']:.3f} ± {s['pr_auc_std']:.3f}  "
                  f"ROC-AUC {s['roc_auc_mean']:.3f}  P@20w {s['precision_at_20w_mean']:.3f}")
        return df


def main() -> None:
    warnings.filterwarnings("ignore")
    params = load_params()
    seed, mp = params["seed"], params["models"]
    r = Runner(params)
    folds_info = [{k: v for k, v in f.items() if k not in ("train", "val")}
                  | {"n_train": len(f["train"]), "n_val": len(f["val"]),
                     "val_success_share": float(r.y[f["val"]].mean())} for f in r.folds]

    # --- п. 5: базовые решения ---
    r.run("Константа (доля успешных)", "baseline", desc="score = 0",
          score_fn=lambda X: np.zeros(len(X)))
    r.run("Правило: у издателя был успешный релиз", "baseline", desc="score = pub_had_hit",
          score_fn=lambda X: (X["pub_n_hit"].values > 0).astype(float))

    # --- п. 6: логистическая регрессия, путь регуляризации L1 и L2 ---
    path_rows = []
    full_X, full_y = r.games, r.y
    for penalty in ["l2", "l1"]:
        for C in mp["logreg_C_grid"]:
            pipe = zoo.make_pipeline(zoo.logreg(C, penalty, seed=seed), params)
            df = r.run(f"logreg_{penalty}_C{C}", "logreg", pipe, record=False)
            fitted = zoo.make_pipeline(zoo.logreg(C, penalty, seed=seed), params).fit(full_X, full_y)
            coef = fitted.named_steps["model"].coef_.ravel()
            path_rows.append({"penalty": penalty, "C": C, "pr_auc": df["pr_auc"].mean(),
                              "pr_auc_std": df["pr_auc"].std(ddof=0),
                              "nonzero": int((np.abs(coef) > 1e-8).sum()), "n_features": len(coef)})
    path = pd.DataFrame(path_rows)
    path.to_csv(REPORTS / "reg_path.csv", index=False)
    best_l2 = path[path.penalty == "l2"].sort_values("pr_auc", ascending=False).iloc[0]
    best_l1 = path[path.penalty == "l1"].sort_values("pr_auc", ascending=False).iloc[0]
    r.run(f"Логистическая регрессия L2 (C = {num(best_l2.C)})", "logreg",
          zoo.make_pipeline(zoo.logreg(best_l2.C, "l2", seed=seed), params), desc=f"l2, C={best_l2.C:g}")
    r.run(f"Логистическая регрессия L1 (C = {num(best_l1.C)})", "logreg",
          zoo.make_pipeline(zoo.logreg(best_l1.C, "l1", seed=seed), params), desc=f"l1, C={best_l1.C:g}")

    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    for pen, mk in [("l2", "o"), ("l1", "s")]:
        p = path[path.penalty == pen]
        ax1.plot(p.C, p.pr_auc, marker=mk, label=f"PR-AUC, {pen.upper()}")
    ax1.set_xscale("log")
    ax1.set_xlabel("C (меньше C — сильнее регуляризация), логарифмическая шкала")
    ax1.set_ylabel("PR-AUC, среднее по 5 блокам")
    ax2 = ax1.twinx()
    p = path[path.penalty == "l1"]
    ax2.plot(p.C, p.nonzero, color="gray", linestyle="--", marker="^", label="Ненулевых коэффициентов, L1")
    ax2.set_ylabel("Число ненулевых коэффициентов, шт.")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right")
    ax1.grid(alpha=0.3)
    ax1.yaxis.set_major_formatter(COMMA)
    ax1.xaxis.set_major_formatter(COMMA)
    fig.tight_layout()
    fig.savefig(FIGURES / "reg_path.png", dpi=200)
    plt.close(fig)

    # Коэффициенты L2-модели на всей исходной части (признаки стандартизованы).
    fitted = zoo.make_pipeline(zoo.logreg(best_l2.C, "l2", seed=seed), params).fit(full_X, full_y)
    names = fitted.named_steps["features"].get_feature_names_out()
    coefs = pd.DataFrame({"feature": names, "coef": fitted.named_steps["model"].coef_.ravel()})
    coefs["abs"] = coefs["coef"].abs()
    coefs.sort_values("abs", ascending=False).drop(columns="abs").to_csv(
        REPORTS / "logreg_coefs.csv", index=False)

    # --- п. 7: модели классификации ---
    r.run(f"k ближайших соседей (k = {mp['knn_k']})", "knn",
          zoo.make_pipeline(zoo.knn(mp["knn_k"]), params), desc=f"k={mp['knn_k']}")
    r.run("Наивный байесовский (гауссовский)", "naive_bayes",
          zoo.make_pipeline(zoo.naive_bayes(), params), desc="GaussianNB")
    depth_rows = []
    for depth in mp["tree_depths"] + [None]:
        df = r.run(f"tree_d{depth}", "tree", zoo.make_pipeline(zoo.tree(depth, seed), params, scale=False),
                   with_train=True, record=False)
        depth_rows.append({"depth": depth if depth else "без ограничения",
                           "val_pr_auc": df["pr_auc"].mean(), "train_pr_auc": df["train_pr_auc"].mean()})
    depth_df = pd.DataFrame(depth_rows)
    depth_df.to_csv(REPORTS / "tree_depth.csv", index=False)
    lim = depth_df[depth_df.depth != "без ограничения"]
    best_depth = int(lim.sort_values("val_pr_auc", ascending=False).iloc[0].depth)
    r.run(f"Дерево решений (глубина {best_depth})", "tree",
          zoo.make_pipeline(zoo.tree(best_depth, seed), params, scale=False), desc=f"max_depth={best_depth}")
    r.run("Дерево решений без ограничения глубины", "tree",
          zoo.make_pipeline(zoo.tree(None, seed), params, scale=False), desc="max_depth=None")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = lim.depth.astype(int)
    ax.plot(x, lim.train_pr_auc, marker="o", label="Обучающие блоки")
    ax.plot(x, lim.val_pr_auc, marker="s", label="Проверочные блоки")
    ax.axvline(best_depth, color="gray", linestyle=":", label=f"Лучшая глубина: {best_depth}")
    ax.set_xlabel("Максимальная глубина дерева")
    ax.set_ylabel("PR-AUC, среднее по 5 блокам")
    ax.set_xticks(x)
    ax.yaxis.set_major_formatter(COMMA)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "tree_depth.png", dpi=200)
    plt.close(fig)

    # --- п. 7: ансамбли ---
    r.run(f"Случайный лес ({mp['rf_trees']} деревьев)", "random_forest",
          zoo.make_pipeline(zoo.random_forest(mp["rf_trees"], seed), params, scale=False),
          desc=f"n_estimators={mp['rf_trees']}, min_samples_leaf=5")
    r.run("Градиентный бустинг LightGBM (по умолчанию)", "lightgbm",
          zoo.make_pipeline(zoo.lightgbm(seed), params, scale=False), desc="defaults")

    # --- п. 7: польза новых признаков ---
    for label, fam, model, scale in [
            (f"Логистическая регрессия L2 (C = {num(best_l2.C)})", "logreg", zoo.logreg(best_l2.C, "l2", seed=seed), True),
            ("Градиентный бустинг LightGBM (по умолчанию)", "lightgbm", zoo.lightgbm(seed), False)]:
        r.run(label + ", базовые признаки", fam, zoo.make_pipeline(model, params, BASE, scale),
              feature_set=BASE, desc="base features")
        r.run(label + ", новые признаки без истории издателя", fam,
              zoo.make_pipeline(model, params, NO_HISTORY, scale),
              feature_set=NO_HISTORY, desc="full features without publisher/developer history")

    summary = pd.DataFrame(r.summary)
    summary.to_csv(REPORTS / "experiments.csv", index=False, float_format="%.4f")
    pd.concat(r.fold_rows).to_csv(REPORTS / "experiments_folds.csv", index=False, float_format="%.4f")
    with open(REPORTS / "cv_folds.json", "w", encoding="utf-8") as f:
        json.dump(folds_info, f, ensure_ascii=False, indent=1)
    with open(REPORTS / "timing_models.json", "w", encoding="utf-8") as f:
        json.dump(r.timing, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
