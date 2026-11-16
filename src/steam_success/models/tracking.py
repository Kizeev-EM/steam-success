"""Учёт экспериментов: MLflow (локальная база sqlite) и сведения о версии."""
import os
import subprocess

import yaml

from steam_success.config import ROOT

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def data_version() -> str:
    """Хеш исходной части из dvc.lock — версия данных, на которых шёл эксперимент."""
    try:
        lock = yaml.safe_load(open(ROOT / "dvc.lock", encoding="utf-8"))
        for out in lock["stages"]["split"]["outs"]:
            if out["path"] == "data/interim/initial.csv":
                return out["md5"]
    except Exception:
        pass
    return "unknown"


def start(params: dict) -> None:
    m = params["mlflow"]
    uri = m["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + str(ROOT / uri.removeprefix("sqlite:///"))
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(m["experiment"])


def log_experiment(name: str, run_params: dict, folds, summary: dict, stage: str) -> None:
    """Одна запись MLflow на эксперимент: параметры, метрики по блокам и средние."""
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"stage": stage, "git_commit": git_commit(), "data_md5": data_version()})
        mlflow.log_params({k: str(v) for k, v in run_params.items()})
        for _, row in folds.iterrows():
            for metric in ["pr_auc", "roc_auc", "pr_auc_lift"]:
                mlflow.log_metric(f"{metric}_fold", float(row[metric]), step=int(row["fold"]))
        mlflow.log_metrics({k: float(v) for k, v in summary.items()
                            if isinstance(v, (int, float))})
