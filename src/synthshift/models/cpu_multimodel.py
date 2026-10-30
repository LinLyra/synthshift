"""Fixed CPU model configurations for the Phase-A ranking audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


@dataclass(frozen=True)
class ModelRun:
    configuration: str
    seed: int
    parameter_name: str
    parameter_values: tuple[float | int, ...]


def feature_indices(configuration: str) -> np.ndarray:
    """Return fixed all/accelerometer/gyroscope statistical-feature indices."""

    if configuration not in {"logistic_acc_only", "logistic_gyro_only"}:
        return np.arange(78)
    acceleration = configuration == "logistic_acc_only"
    indices = []
    for statistic in range(9):
        start = statistic * 6
        indices.extend(range(start, start + 3) if acceleration else range(start + 3, start + 6))
    for statistic in range(9):
        indices.append(54 + statistic * 2 + (0 if acceleration else 1))
    indices.extend(range(72, 75) if acceleration else range(75, 78))
    return np.asarray(indices, dtype=np.int64)


def model_runs() -> list[ModelRun]:
    return [
        ModelRun("logistic_all", 0, "C", (0.1, 1.0, 10.0)),
        ModelRun("linear_svm", 0, "C", (0.1, 1.0, 10.0)),
        ModelRun("rbf_svm", 0, "C", (1.0, 10.0)),
        ModelRun("logistic_acc_only", 0, "C", (0.1, 1.0, 10.0)),
        ModelRun("logistic_gyro_only", 0, "C", (0.1, 1.0, 10.0)),
        *[
            ModelRun(configuration, seed, "min_samples_leaf", (1, 5))
            for configuration in ("random_forest", "extra_trees")
            for seed in (0, 1, 2)
        ],
    ]


def build_model(configuration: str, parameter: float | int, seed: int) -> object:
    if configuration.startswith("logistic_"):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(parameter), class_weight="balanced", max_iter=2_000, random_state=seed
            ),
        )
    if configuration == "linear_svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(C=float(parameter), class_weight="balanced", dual=False, random_state=seed),
        )
    if configuration == "rbf_svm":
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=float(parameter),
                gamma="scale",
                class_weight="balanced",
                cache_size=1024,
                random_state=seed,
            ),
        )
    common = {
        "n_estimators": 400,
        "max_features": "sqrt",
        "min_samples_leaf": int(parameter),
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": -1,
    }
    if configuration == "random_forest":
        return RandomForestClassifier(**common)
    if configuration == "extra_trees":
        return ExtraTreesClassifier(**common)
    raise ValueError(f"Unknown configuration: {configuration}")
