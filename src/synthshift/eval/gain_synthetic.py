"""Exploratory accelerometer-gain synthetic-to-real experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synthshift.data.features import extract_features
from synthshift.devices.gain import apply_accelerometer_gain, estimate_gain_profile, gain_ratio
from synthshift.eval.affine_synthetic import _raw_windows, _score
from synthshift.eval.noise_synthetic import _fit
from synthshift.models.statistical_baseline import _model


def run_gain_experiment(
    features_path: Path,
    split_path: Path,
    manifest_path: Path,
    segment_root: Path,
    output_path: Path,
    *,
    comparison_paths: dict[str, Path] | None = None,
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
            profile_cache[(fold["name"], device)] = estimate_gain_profile(
                manifest_path,
                segment_root,
                fold["target_calibration_users"],
                device,
            )

        for source in devices:
            print(f"{fold['name']} {source}", flush=True)
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
            source_gain = float(profile_cache[(fold["name"], source)]["gain"])

            for target in devices:
                target_test = test & (table["devices"] == target)
                real_score = _score(
                    model, table["features"][target_test], table["labels"][target_test]
                )
                ratio = gain_ratio(
                    source_gain, float(profile_cache[(fold["name"], target)]["gain"])
                )
                if ratio == 1.0:
                    synthetic_score = source_score
                else:
                    transformed = np.stack(
                        [apply_accelerometer_gain(window, ratio) for window in source_windows]
                    )
                    features = np.stack([extract_features(window) for window in transformed])
                    synthetic_score = _score(model, features, source_truth)
                results.append(
                    {
                        "fold": fold["name"],
                        "source_device": source,
                        "target_device": target,
                        "source_macro_f1": round(source_score, 6),
                        "real_target_macro_f1": round(real_score, 6),
                        "synthetic_target_macro_f1": round(float(synthetic_score), 6),
                        "real_degradation": round(source_score - real_score, 6),
                        "synthetic_degradation": round(source_score - float(synthetic_score), 6),
                        "degradation_error": round(
                            (source_score - float(synthetic_score)) - (source_score - real_score),
                            6,
                        ),
                        "gain_ratio": ratio,
                        "calibration_users": fold["target_calibration_users"],
                    }
                )

    shifted = [row for row in results if row["source_device"] != row["target_device"]]
    real = np.asarray([row["real_degradation"] for row in shifted], dtype=np.float64)
    synthetic = np.asarray([row["synthetic_degradation"] for row in shifted], dtype=np.float64)
    summary = {
        "off_diagonal_evaluations": len(shifted),
        **_fit(real, synthetic, "gain"),
        "null_degradation_mae": float(np.mean(np.abs(real))),
        "max_abs_gain_ratio_error": float(
            np.max(np.abs(np.asarray([row["gain_ratio"] for row in shifted]) - 1.0))
        ),
    }
    for name, path in (comparison_paths or {}).items():
        prior = json.loads(path.read_text(encoding="utf-8"))
        prior_rows = {
            (row["fold"], row["source_device"], row["target_device"]): row
            for row in prior["results"]
            if row["source_device"] != row["target_device"]
        }
        prior_degradation = np.asarray(
            [
                prior_rows[(row["fold"], row["source_device"], row["target_device"])][
                    "synthetic_degradation"
                ]
                for row in shifted
            ],
            dtype=np.float64,
        )
        summary.update(_fit(real, prior_degradation, name))
    payload = {
        "method": (
            "unlabelled accelerometer gain from median still acceleration magnitude; "
            "gyroscope is unchanged and no cross-phone time alignment is used"
        ),
        "summary": summary,
        "profiles": list(profile_cache.values()),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
