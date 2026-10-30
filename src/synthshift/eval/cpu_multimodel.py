"""Frozen CPU multi-model gain and ranking audit."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import f1_score

from synthshift.data.features import extract_features
from synthshift.devices.gain import apply_accelerometer_gain, estimate_gain_profile, gain_ratio
from synthshift.eval.affine_synthetic import _raw_windows
from synthshift.eval.gyro_bias_factor import _extended_fit
from synthshift.models.cpu_multimodel import build_model, feature_indices, model_runs


def _pooled_score(model: object, features: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(labels, model.predict(features), average="macro", zero_division=0))


def _subject_macro_score(
    model: object, features: np.ndarray, labels: np.ndarray, users: np.ndarray
) -> float:
    scores = []
    for user in sorted(np.unique(users)):
        selected = users == user
        scores.append(
            f1_score(
                labels[selected],
                model.predict(features[selected]),
                average="macro",
                zero_division=0,
            )
        )
    return float(np.mean(scores))


def _pairwise_accuracy(real: np.ndarray, synthetic: np.ndarray, *, lower_better: bool) -> float:
    points = 0.0
    pairs = 0
    direction = -1.0 if lower_better else 1.0
    for first, second in combinations(range(len(real)), 2):
        real_sign = np.sign(direction * (real[first] - real[second]))
        synthetic_sign = np.sign(direction * (synthetic[first] - synthetic[second]))
        pairs += 1
        if real_sign == synthetic_sign:
            points += 1.0
        elif real_sign == 0 or synthetic_sign == 0:
            points += 0.5
    return points / pairs


def _ranking_metrics(real: np.ndarray, synthetic: np.ndarray, *, lower_better: bool) -> dict:
    order_real = np.argsort(real if lower_better else -real)
    order_synthetic = np.argsort(synthetic if lower_better else -synthetic)
    return {
        "kendall_tau_b": float(kendalltau(real, synthetic).statistic),
        "spearman_r": float(spearmanr(real, synthetic).statistic),
        "top2_overlap": float(len(set(order_real[:2]) & set(order_synthetic[:2])) / 2.0),
        "pairwise_ordering_accuracy": float(
            _pairwise_accuracy(real, synthetic, lower_better=lower_better)
        ),
    }


def run_cpu_multimodel_audit(
    features_path: Path,
    split_path: Path,
    manifest_path: Path,
    segment_root: Path,
    output_path: Path,
) -> dict[str, object]:
    with np.load(features_path) as bundle:
        table = {name: bundle[name] for name in bundle.files}
    config = json.loads(split_path.read_text(encoding="utf-8"))
    devices = config["primary_devices"]
    profiles = {}
    results = []
    runs = model_runs()

    for fold in config["folds"]:
        train = np.isin(table["users"], fold["train_users"])
        validation = np.isin(table["users"], fold["validation_users"])
        test = np.isin(table["users"], fold["test_users"])
        for device in devices:
            key = (fold["name"], device)
            profiles[key] = estimate_gain_profile(
                manifest_path, segment_root, fold["target_calibration_users"], device
            )
            if profiles[key]["usable_users"] != fold["target_calibration_users"]:
                raise ValueError(f"Gain profile lacks both users: {key}")

        for source in devices:
            print(f"{fold['name']} {source}", flush=True)
            source_mask = table["devices"] == source
            source_indices = np.flatnonzero(test & source_mask)
            source_windows = _raw_windows(table, source_indices, segment_root)
            synthetic_cache = {}
            for target in devices:
                ratio = gain_ratio(
                    float(profiles[(fold["name"], source)]["gain"]),
                    float(profiles[(fold["name"], target)]["gain"]),
                )
                if source == target:
                    synthetic_cache[target] = table["features"][source_indices]
                else:
                    transformed = np.stack(
                        [apply_accelerometer_gain(window, ratio) for window in source_windows]
                    )
                    synthetic_cache[target] = np.stack(
                        [extract_features(window) for window in transformed]
                    )

            for run in runs:
                columns = feature_indices(run.configuration)
                best_parameter, best_validation = None, -1.0
                for parameter in run.parameter_values:
                    candidate = build_model(run.configuration, parameter, run.seed)
                    candidate.fit(
                        table["features"][train & source_mask][:, columns],
                        table["labels"][train & source_mask],
                    )
                    score = _pooled_score(
                        candidate,
                        table["features"][validation & source_mask][:, columns],
                        table["labels"][validation & source_mask],
                    )
                    if score > best_validation:
                        best_parameter, best_validation = parameter, score
                model = build_model(run.configuration, best_parameter, run.seed)
                fit = (train | validation) & source_mask
                model.fit(table["features"][fit][:, columns], table["labels"][fit])
                source_features = table["features"][source_indices][:, columns]
                source_labels = table["labels"][source_indices]
                source_users = table["users"][source_indices]
                source_pooled = _pooled_score(model, source_features, source_labels)
                source_subject = _subject_macro_score(
                    model, source_features, source_labels, source_users
                )
                for target in devices:
                    target_mask = test & (table["devices"] == target)
                    target_features = table["features"][target_mask][:, columns]
                    target_labels = table["labels"][target_mask]
                    target_users = table["users"][target_mask]
                    synthetic_features = synthetic_cache[target][:, columns]
                    real_pooled = _pooled_score(model, target_features, target_labels)
                    synthetic_pooled = _pooled_score(model, synthetic_features, source_labels)
                    real_subject = _subject_macro_score(
                        model, target_features, target_labels, target_users
                    )
                    synthetic_subject = _subject_macro_score(
                        model, synthetic_features, source_labels, source_users
                    )
                    results.append(
                        {
                            "fold": fold["name"],
                            "source_device": source,
                            "target_device": target,
                            "configuration": run.configuration,
                            "seed": run.seed,
                            "selected_parameter": best_parameter,
                            "validation_macro_f1": best_validation,
                            "source_pooled_macro_f1": source_pooled,
                            "real_target_pooled_macro_f1": real_pooled,
                            "synthetic_target_pooled_macro_f1": synthetic_pooled,
                            "real_pooled_degradation": source_pooled - real_pooled,
                            "synthetic_pooled_degradation": source_pooled - synthetic_pooled,
                            "source_subject_macro_f1": source_subject,
                            "real_target_subject_macro_f1": real_subject,
                            "synthetic_target_subject_macro_f1": synthetic_subject,
                            "real_subject_degradation": source_subject - real_subject,
                            "synthetic_subject_degradation": source_subject - synthetic_subject,
                        }
                    )

    aggregate = _aggregate_seeds(results)
    payload = {
        "protocol_commit": "276298a",
        "method": "fixed seven-configuration CPU gain and ranking audit",
        "summary": _summarize(aggregate),
        "profiles": list(profiles.values()),
        "seed_results": results,
        "aggregate_results": aggregate,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _aggregate_seeds(results: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = {}
    for row in results:
        key = (row["fold"], row["source_device"], row["target_device"], row["configuration"])
        grouped.setdefault(key, []).append(row)
    fields = [
        "source_pooled_macro_f1", "real_target_pooled_macro_f1",
        "synthetic_target_pooled_macro_f1", "real_pooled_degradation",
        "synthetic_pooled_degradation", "source_subject_macro_f1",
        "real_target_subject_macro_f1", "synthetic_target_subject_macro_f1",
        "real_subject_degradation", "synthetic_subject_degradation",
    ]
    aggregate = []
    for key, rows in grouped.items():
        item = {
            "fold": key[0], "source_device": key[1], "target_device": key[2],
            "configuration": key[3], "seed_count": len(rows),
        }
        item.update({field: float(np.mean([row[field] for row in rows])) for field in fields})
        aggregate.append(item)
    return aggregate


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    configurations = sorted({row["configuration"] for row in rows})
    off_diagonal = [row for row in rows if row["source_device"] != row["target_device"]]
    by_configuration = {}
    for configuration in configurations:
        selected = [row for row in off_diagonal if row["configuration"] == configuration]
        entry = {}
        for score in ("pooled", "subject"):
            real = np.asarray([row[f"real_{score}_degradation"] for row in selected])
            synthetic = np.asarray([row[f"synthetic_{score}_degradation"] for row in selected])
            entry[score] = _extended_fit(real, synthetic, f"gain_{score}")
        by_configuration[configuration] = entry

    ranking_rows = []
    cell_keys = sorted(
        {(row["fold"], row["source_device"], row["target_device"]) for row in off_diagonal}
    )
    for key in cell_keys:
        selected = sorted(
            [
                row for row in off_diagonal
                if (row["fold"], row["source_device"], row["target_device"]) == key
            ],
            key=lambda row: row["configuration"],
        )
        item = {"fold": key[0], "source_device": key[1], "target_device": key[2]}
        for score in ("pooled", "subject"):
            real_degradation = np.asarray([row[f"real_{score}_degradation"] for row in selected])
            synthetic_degradation = np.asarray(
                [row[f"synthetic_{score}_degradation"] for row in selected]
            )
            real_target = np.asarray([row[f"real_target_{score}_macro_f1"] for row in selected])
            synthetic_target = np.asarray(
                [row[f"synthetic_target_{score}_macro_f1"] for row in selected]
            )
            item[f"{score}_robustness"] = _ranking_metrics(
                real_degradation, synthetic_degradation, lower_better=True
            )
            item[f"{score}_performance"] = _ranking_metrics(
                real_target, synthetic_target, lower_better=False
            )
        ranking_rows.append(item)
    ranking_mean = {}
    for score in ("pooled", "subject"):
        for target in ("robustness", "performance"):
            name = f"{score}_{target}"
            ranking_mean[name] = {
                metric: float(np.nanmean([row[name][metric] for row in ranking_rows]))
                for metric in (
                    "kendall_tau_b", "spearman_r", "top2_overlap",
                    "pairwise_ordering_accuracy",
                )
            }
    return {
        "configurations": configurations,
        "off_diagonal_cells_per_configuration": 60,
        "by_configuration": by_configuration,
        "ranking_mean": ranking_mean,
        "ranking_cells": ranking_rows,
    }
