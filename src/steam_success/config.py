"""Загрузка параметров и общие пути проекта."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "configs" / "params.yaml"

INTERIM = ROOT / "data" / "interim"
INCOMING = ROOT / "data" / "incoming"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"


def load_params(path: Path = PARAMS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
