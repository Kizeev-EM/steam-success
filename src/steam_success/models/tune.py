"""Стадия tune: подбор гиперпараметров лучших моделей (ПР 3, п. 8).

Перебор по сетке из configs/params.yaml; каждый набор оценивается той же
пятиблочной временной кросс-валидацией, что и в experiments. Помимо
гиперпараметров моделей перебирается вес недавних лет (half-life): по
итогам гипотезы Г3 ПР 2 зависимость успеха от признаков меняется со
временем, и свежие игры могут быть полезнее старых.
"""
import itertools
import json
import warnings

import pandas as pd

from steam_success.config import REPORTS, load_params
from steam_success.features.build import load_initial
from steam_success.models import tracking, zoo
from steam_success.models.cv import cross_validate, time_folds


def grid(space: dict):
    keys = list(space)
    for values in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, values))


def build(family: str, hp: dict, params: dict):
    seed = params["seed"]
    if family == "lightgbm":
        model = zoo.lightgbm(seed, learning_rate=hp["learning_rate"], num_leaves=hp["num_leaves"],
                             min_child_samples=hp["min_child_samples"], n_estimators=hp["n_estimators"])
        return zoo.make_pipeline(model, params, scale=False)
    if family == "random_forest":
        model = zoo.random_forest(params["models"]["rf_trees"], seed)
        model.set_params(min_samples_leaf=hp["min_samples_leaf"], max_features=hp["max_features"])
        return zoo.make_pipeline(model, params, scale=False)
    if family == "logreg":
        return zoo.make_pipeline(zoo.logreg(hp["C"], "l2", hp["class_weight"], seed), params)
    raise ValueError(family)


def main() -> None:
    warnings.filterwarnings("ignore")
    params = load_params()
    cv = params["cv"]
    games, _, _ = load_initial(params)
    y = games["y"].values
    folds = time_folds(games["release_date"], cv)
    tracking.start(params)
    best = {}
    for family, space in params["models"]["tune"].items():
        rows = []
        for hp in grid(space):
            df = cross_validate(build(family, hp, params), games, y, folds, cv["top_k"],
                                half_life=hp["recency_half_life_years"])
            rows.append(hp | {"pr_auc_mean": df["pr_auc"].mean(), "pr_auc_std": df["pr_auc"].std(ddof=0),
                              "roc_auc_mean": df["roc_auc"].mean(),
                              "precision_at_20w_mean": df["precision_at_20w"].mean(),
                              "recall_at_20w_mean": df["recall_at_20w"].mean(),
                              "pr_auc_lift_mean": df["pr_auc_lift"].mean(),
                              "fit_time": df["fit_time"].mean()})
        res = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False, kind="mergesort")
        res.drop(columns="fit_time").to_csv(REPORTS / f"tuning_{family}.csv", index=False,
                                            float_format="%.4f")
        top = res.iloc[0].to_dict()
        hp = list(grid(space))[res.index[0]]          # исходные типы значений сетки
        best[family] = {"params": hp, "n_combinations": len(res),
                        "pr_auc_mean": round(top["pr_auc_mean"], 4),
                        "pr_auc_std": round(top["pr_auc_std"], 4)}
        df = cross_validate(build(family, hp, params), games, y, folds, cv["top_k"],
                            half_life=hp["recency_half_life_years"])
        tracking.log_experiment(f"{family} (после подбора)", hp, df,
                                {"pr_auc_mean": df["pr_auc"].mean()}, stage="tune")
        print(family, len(res), "наборов; лучший:", hp, round(top["pr_auc_mean"], 4))
    with open(REPORTS / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
