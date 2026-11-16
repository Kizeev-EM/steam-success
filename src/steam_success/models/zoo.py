"""Модели и пайплайн «признаки -> масштабирование -> модель» (ПР 3, пп. 5–7)."""
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from steam_success.features.build import FULL, CardFeaturizer

# Чем меньше число, тем модель проще (объём модели и время расчёта): используется
# в правиле выбора. Бустинг из неглубоких деревьев компактнее леса из глубоких.
COMPLEXITY = {"baseline": 0, "logreg": 1, "naive_bayes": 1, "tree": 2, "knn": 3,
              "lightgbm": 4, "random_forest": 5}


def make_pipeline(model, params: dict, feature_set: str = FULL, scale: bool = True) -> Pipeline:
    f = params["features"]
    return Pipeline([
        ("features", CardFeaturizer(feature_set, f["min_count"], f["ach_cap"],
                                    tuple(f["online_categories"]))),
        ("scale", StandardScaler() if scale else "passthrough"),
        ("model", model),
    ])


def logreg(C=1.0, penalty="l2", class_weight="balanced", seed=42):
    # liblinear поддерживает и L1, и L2 и детерминирован при фиксированном seed.
    return LogisticRegression(C=C, l1_ratio=1.0 if penalty == "l1" else 0.0,
                              solver="liblinear", class_weight=class_weight,
                              max_iter=2000, random_state=seed)


def knn(k=25):
    return KNeighborsClassifier(n_neighbors=k)


def naive_bayes():
    return GaussianNB()


def tree(depth=None, seed=42):
    return DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=seed)


def random_forest(n=300, seed=42):
    return RandomForestClassifier(n_estimators=n, min_samples_leaf=5, class_weight="balanced",
                                  n_jobs=-1, random_state=seed)


def lightgbm(seed=42, **kw):
    base = dict(class_weight="balanced", random_state=seed, verbose=-1,
                deterministic=True, force_row_wise=True, n_jobs=2)
    base.update(kw)
    return LGBMClassifier(**base)
