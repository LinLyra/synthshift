"""Exploratory calibrated-affine synthetic-to-real experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import f1_score

from synthshift.data.features import extract_features
from synthshift.devices.profile import affine_parameters, apply_affine, estimate_device_profile
from synthshift.models.statistical_baseline import _model


def _raw_windows(
    table: dict[str, np.ndarray], indices: np.ndarray, segment_root: Path
) -> np.ndarray:
    windows = []
    cached_file = None
    cached_values = None
    for index in indices:
        filename = str(table["segment_files"][index])
        if filename != cached_file:
            with np.load(segment_root / filename) as bundle:
                cached_values = bundle["values"]
            cached_file = filename
        offset = int(table["offsets"][index])
        windows.append(cached_values[offset : offset + 250])
    return np.asarray(windows)


def _score(model: object, features: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(labels, model.predict(features), average="macro", zero_division=0))


def run_affine_experiment(
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
    profile_cache = {}
    results = []

    for fold in config["folds"]:
        train = np.isin(table["users"], fold["train_users"])
        validation = np.isin(table["users"], fold["validation_users"])
        test = np.isin(table["users"], fold["test_users"])
        for device in devices:
            profile_cache[(fold["name"], device)] = estimate_device_profile(
                manifest_path,
                segment_root,
                fold["target_calibration_users"],
                device,
            )

        for source in devices:
            source_mask = table["devices"] == source
            best_c, best_validation = None, -1.0
            for c_value in (0.1, 1.0, 10.0):
                candidate = _model(c_value)
                candidate.fit(
                    table["features"][train & source_mask], table["labels"][train & source_mask]
                )
                score = _score(
                    candidate,
                    table["features"][validation & source_mask],
                    table["labels"][validation & source_mask],
                )
                if score > best_validation:
                    best_c, best_validation = c_value, score

            model = _model(float(best_c))
            fit = (train | validation) & source_mask
            model.fit(table["features"][fit], table["labels"][fit])
            source_test_indices = np.flatnonzero(test & source_mask)
            source_truth = table["labels"][source_test_indices]
            source_score = _score(model, table["features"][source_test_indices], source_truth)
            source_windows = _raw_windows(table, source_test_indices, segment_root)

            for target in devices:
                target_test = test & (table["devices"] == target)
                real_score = _score(
                    model, table["features"][target_test], table["labels"][target_test]
                )
                scale, offset = affine_parameters(
                    profile_cache[(fold["name"], source)],
                    profile_cache[(fold["name"], target)],
                )
                transformed = apply_affine(source_windows, scale, offset)
                synthetic_features = np.stack([extract_features(window) for window in transformed])
                synthetic_score = _score(model, synthetic_features, source_truth)
                results.append(
                    {
                        "fold": fold["name"],
                        "source_device": source,
                        "target_device": target,
                        "source_macro_f1": round(source_score, 6),
                        "real_target_macro_f1": round(real_score, 6),
                        "synthetic_target_macro_f1": round(synthetic_score, 6),
                        "real_degradation": round(source_score - real_score, 6),
                        "synthetic_degradation": round(source_score - synthetic_score, 6),
                        "degradation_error": round(synthetic_score - real_score, 6),
                        "scale": scale.tolist(),
                        "offset": offset.tolist(),
                        "calibration_users": fold["target_calibration_users"],
                    }
                )

    shifted = [row for row in results if row["source_device"] != row["target_device"]]
    real = np.asarray([row["real_degradation"] for row in shifted])
    synthetic = np.asarray([row["synthetic_degradation"] for row in shifted])
    rank_correlations = []
    for fold in config["folds"]:
        for source in devices:
            group = [
                row
                for row in results
                if row["fold"] == fold["name"] and row["source_device"] == source
            ]
            rank_correlations.append(
                float(
                    kendalltau(
                        [row["real_target_macro_f1"] for row in group],
                        [row["synthetic_target_macro_f1"] for row in group],
                    ).statistic
                )
            )
    summary = {
        "off_diagonal_evaluations": len(shifted),
        "degradation_mae": float(np.mean(np.abs(synthetic - real))),
        "degradation_rmse": float(np.sqrt(np.mean(np.square(synthetic - real)))),
        "pearson_r": float(pearsonr(real, synthetic).statistic),
        "spearman_r": float(spearmanr(real, synthetic).statistic),
        "mean_target_ranking_kendall_tau": float(np.nanmean(rank_correlations)),
        "ranking_scope": "target-device ranking for one model family; not RQ2 model ranking",
    }
    payload = {
        "method": "unlabelled per-channel median/IQR affine calibration",
        "summary": summary,
        "profiles": list(profile_cache.values()),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
