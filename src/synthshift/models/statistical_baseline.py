"""Subject-disjoint statistical-feature baseline evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _model(c_value: float) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, class_weight="balanced", max_iter=2_000, random_state=0),
    )


def run_baseline(
    features_path: Path, split_path: Path, output_path: Path
) -> list[dict[str, object]]:
    with np.load(features_path) as table:
        features = table["features"]
        users = table["users"]
        devices = table["devices"]
        labels = table["labels"]
    config = json.loads(split_path.read_text(encoding="utf-8"))
    primary_devices = config["primary_devices"]
    label_order = sorted(np.unique(labels).tolist())
    results = []

    for fold in config["folds"]:
        train_users = np.isin(users, fold["train_users"])
        validation_users = np.isin(users, fold["validation_users"])
        test_users = np.isin(users, fold["test_users"])
        for source_device in primary_devices:
            source = devices == source_device
            best_c = None
            best_validation = -1.0
            for c_value in (0.1, 1.0, 10.0):
                model = _model(c_value)
                model.fit(features[train_users & source], labels[train_users & source])
                prediction = model.predict(features[validation_users & source])
                score = f1_score(
                    labels[validation_users & source], prediction, average="macro", zero_division=0
                )
                if score > best_validation:
                    best_c, best_validation = c_value, score

            fit_mask = (train_users | validation_users) & source
            model = _model(float(best_c))
            model.fit(features[fit_mask], labels[fit_mask])
            for target_device in primary_devices:
                evaluation = test_users & (devices == target_device)
                truth = labels[evaluation]
                prediction = model.predict(features[evaluation])
                results.append(
                    {
                        "fold": fold["name"],
                        "source_device": source_device,
                        "target_device": target_device,
                        "model": "standardized_logistic_regression",
                        "selected_c": best_c,
                        "validation_macro_f1": round(best_validation, 6),
                        "test_windows": int(np.sum(evaluation)),
                        "macro_f1": round(f1_score(truth, prediction, average="macro"), 6),
                        "balanced_accuracy": round(balanced_accuracy_score(truth, prediction), 6),
                        "confusion_matrix": confusion_matrix(
                            truth, prediction, labels=label_order
                        ).tolist(),
                        "label_order": label_order,
                    }
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
    return results
