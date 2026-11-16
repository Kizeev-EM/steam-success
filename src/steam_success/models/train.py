"""Стадия train: сводная таблица, выбор и сохранение итоговой модели (ПР 3, п. 9).

Правило выбора (п. 1 отчёта): наибольшее среднее PR-AUC по пяти блокам;
если другая модель уступает лучшей меньше чем на selection_tolerance,
выбирается более простая. Итоговая модель обучается на всей модельной
выборке исходной части и сохраняется в models/model_v1.joblib вместе с
описанием models/model_v1.json; в MLflow она регистрируется как версия
модели steam-success с псевдонимом production.
"""
import json
import os
import warnings

import joblib
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from steam_success.config import FIGURES, MODELS, REPORTS, load_params
from steam_success.features.build import FULL, load_initial
from steam_success.models import tracking, zoo
from steam_success.models.cv import cross_validate, recency_weights, time_folds
from steam_success.models.tune import build

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
COMMA = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
TUNED_NAMES = {"lightgbm": "Градиентный бустинг LightGBM (после подбора)",
               "random_forest": "Случайный лес (после подбора)",
               "logreg": "Логистическая регрессия L2 (после подбора)"}


def summary_table() -> pd.DataFrame:
    exp = pd.read_csv(REPORTS / "experiments.csv")
    best = json.load(open(REPORTS / "best_params.json", encoding="utf-8"))
    rows = []
    for family, b in best.items():
        t = pd.read_csv(REPORTS / f"tuning_{family}.csv").iloc[0]
        rows.append({"experiment": TUNED_NAMES[family], "family": family, "feature_set": FULL,
                     "params": json.dumps(b["params"], ensure_ascii=False),
                     "pr_auc_mean": t.pr_auc_mean, "pr_auc_lift_mean": t.pr_auc_lift_mean,
                     "roc_auc_mean": t.roc_auc_mean, "precision_at_20w_mean": t.precision_at_20w_mean,
                     "recall_at_20w_mean": t.recall_at_20w_mean, "pr_auc_std": t.pr_auc_std})
    return pd.concat([exp, pd.DataFrame(rows)], ignore_index=True)


def choose(summary: pd.DataFrame, tol: float) -> pd.Series:
    cand = summary[(summary.family != "baseline") & (summary.feature_set == FULL)].copy()
    cand["complexity"] = cand.family.map(zoo.COMPLEXITY)
    top = cand.pr_auc_mean.max()
    close = cand[cand.pr_auc_mean >= top - tol]
    return close.sort_values(["complexity", "pr_auc_mean"], ascending=[True, False]).iloc[0]


def final_pipeline(row: pd.Series, params: dict):
    """Пайплайн и вес недавних лет для выбранной строки сводной таблицы."""
    best = json.load(open(REPORTS / "best_params.json", encoding="utf-8"))
    if row.experiment in TUNED_NAMES.values():
        hp = best[row.family]["params"]
        return build(row.family, hp, params), hp.get("recency_half_life_years"), hp
    raise ValueError("Итоговой выбирается модель после подбора гиперпараметров")


def shap_importance(params: dict, games: pd.DataFrame, y: np.ndarray, hp: dict) -> pd.DataFrame:
    """Встроенный отбор признаков: средний |SHAP| бустинга после подбора.

    LightGBM считает значения SHAP сам (TreeSHAP, predict(pred_contrib=True)).
    Модель обучается на всей исходной части, значения — по последнему блоку.
    """
    pipe = build("lightgbm", hp, params)
    w = recency_weights(games["release_date"], hp.get("recency_half_life_years"))
    pipe.fit(games, y, **({"model__sample_weight": w} if w is not None else {}))
    last = time_folds(games["release_date"], params["cv"])[-1]["val"]
    Xf = pipe.named_steps["features"].transform(games.iloc[last])
    contrib = pipe.named_steps["model"].booster_.predict(Xf.values, pred_contrib=True)[:, :-1]
    imp = pd.DataFrame({"feature": Xf.columns, "mean_abs_shap": np.abs(contrib).mean(axis=0)})
    return imp.sort_values("mean_abs_shap", ascending=False, kind="mergesort")


def main() -> None:
    warnings.filterwarnings("ignore")
    params = load_params()
    cv = params["cv"]
    summary = summary_table()
    summary.to_csv(REPORTS / "summary.csv", index=False, float_format="%.4f")
    row = choose(summary, cv["selection_tolerance"])
    print("Итоговая модель:", row.experiment, round(row.pr_auc_mean, 4))

    # Рисунок: сравнение моделей по PR-AUC (полный набор признаков).
    plot = summary[summary.feature_set == FULL].sort_values("pr_auc_mean")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ["#C44E52" if e == row.experiment else ("#999999" if f == "baseline" else "#4C72B0")
              for e, f in zip(plot.experiment, plot.family)]
    ax.barh(plot.experiment, plot.pr_auc_mean, xerr=plot.pr_auc_std, color=colors, capsize=3)
    for i, v in enumerate(plot.pr_auc_mean):
        ax.text(v + 0.003, i - 0.32, f"{v:.3f}".replace(".", ","), fontsize=8)
    ax.set_xlabel("PR-AUC, среднее по 5 блокам (отрезок — стандартное отклонение)")
    ax.xaxis.set_major_formatter(COMMA)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "models_comparison.png", dpi=200)
    plt.close(fig)

    games, _, _ = load_initial(params)
    y = games["y"].values
    pipe, half_life, hp = final_pipeline(row, params)
    w = recency_weights(games["release_date"], half_life)
    pipe.fit(games, y, **({"model__sample_weight": w} if w is not None else {}))

    MODELS.mkdir(exist_ok=True)
    joblib.dump(pipe, MODELS / "model_v1.joblib")
    card = {
        "version": "v1",
        "algorithm": row.experiment,
        "family": row.family,
        "hyperparameters": hp,
        "features": list(pipe.named_steps["features"].get_feature_names_out()),
        "data": {"file": "data/interim/initial.csv", "md5": tracking.data_version(),
                 "period": f"{games.release_date.min().date()} — {games.release_date.max().date()}",
                 "rows": int(len(games)), "success_share": round(float(y.mean()), 4),
                 "selection": "платные игры без жанров-программ"},
        "target": f"нижняя граница owners >= {params['target']['threshold']}",
        "cv": {"scheme": f"{cv['n_folds']} блоков по {cv['fold_months']} мес., зазор {cv['gap_months']} мес.",
               "pr_auc_mean": round(float(row.pr_auc_mean), 4),
               "pr_auc_std": round(float(row.pr_auc_std), 4),
               "roc_auc_mean": round(float(row.roc_auc_mean), 4),
               "precision_at_20w_mean": round(float(row.precision_at_20w_mean), 4)},
        "seed": params["seed"],
        "reproduce": "dvc repro train",
    }
    with open(MODELS / "model_v1.json", "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=1, default=float)

    # Встроенный отбор признаков по SHAP бустинга после подбора.
    best = json.load(open(REPORTS / "best_params.json", encoding="utf-8"))
    imp = shap_importance(params, games, y, best["lightgbm"]["params"])
    imp.to_csv(REPORTS / "shap_importance.csv", index=False, float_format="%.5f")
    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.feature, top.mean_abs_shap, color="#4C72B0")
    ax.set_xlabel("Средний модуль значения SHAP, единицы логарифма шансов")
    ax.xaxis.set_major_formatter(COMMA)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "shap_importance.png", dpi=200)
    plt.close(fig)

    # Регистрация в MLflow: версия модели и псевдоним production.
    tracking.start(params)
    import mlflow
    from mlflow import MlflowClient
    folds = time_folds(games["release_date"], cv)
    df = cross_validate(build(row.family, hp, params), games, y, folds, cv["top_k"], half_life=half_life)
    with mlflow.start_run(run_name="model_v1") as run:
        mlflow.set_tags({"stage": "train", "git_commit": tracking.git_commit(),
                         "data_md5": tracking.data_version()})
        mlflow.log_params({k: str(v) for k, v in hp.items()} | {"algorithm": row.experiment})
        mlflow.log_metrics({"cv_pr_auc_mean": float(df.pr_auc.mean()),
                            "cv_roc_auc_mean": float(df.roc_auc.mean())})
        mlflow.log_artifact(str(MODELS / "model_v1.json"))
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
        client = MlflowClient()
        name = params["mlflow"]["registered_model"]
        mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", name)
        client.set_registered_model_alias(name, "production", mv.version)
    print("MLflow: зарегистрирована версия", mv.version)


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    main()
