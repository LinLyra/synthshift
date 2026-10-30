"""Frozen one-sided s3mini gain-plus-clipping case study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synthshift.data.features import extract_features
from synthshift.devices.clipping import (
    S3MINI_RAIL_M_S2,
    clip_accelerometer,
    clipping_diagnostics,
)
from synthshift.devices.gain import apply_accelerometer_gain, estimate_gain_profile, gain_ratio
from synthshift.eval.affine_synthetic import _raw_windows, _score
from synthshift.eval.gyro_bias_factor import _extended_fit
from synthshift.models.statistical_baseline import _model

TARGET_DEVICE = "s3mini_1"


def run_s3mini_clipping_experiment(
    features_path: Path,
    split_path: Path,
    manifest_path: Path,
    segment_root: Path,
    output_path: Path,
) -> dict[str, object]:
    with np.load(features_path) as bundle:
        table = {name: bundle[name] for name in bundle.files}
    config = json.loads(split_path.read_text(encoding="utf-8"))
    sources = [device for device in config["primary_devices"] if device != TARGET_DEVICE]
    profiles = {}
    results = []
    for fold in config["folds"]:
        train = np.isin(table["users"], fold["train_users"])
        validation = np.isin(table["users"], fold["validation_users"])
        test = np.isin(table["users"], fold["test_users"])
        for device in config["primary_devices"]:
            key = (fold["name"], device)
            profiles[key] = estimate_gain_profile(
                manifest_path, segment_root, fold["target_calibration_users"], device
            )
            if profiles[key]["usable_users"] != fold["target_calibration_users"]:
                raise ValueError(f"Gain profile lacks both calibration users: {key}")

        for source in sources:
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
            source_indices = np.flatnonzero(test & source_mask)
            source_truth = table["labels"][source_indices]
            source_score = _score(model, table["features"][source_indices], source_truth)
            source_windows = _raw_windows(table, source_indices, segment_root)
            ratio = gain_ratio(
                float(profiles[(fold["name"], source)]["gain"]),
                float(profiles[(fold["name"], TARGET_DEVICE)]["gain"]),
            )
            gained = np.stack(
                [apply_accelerometer_gain(window, ratio) for window in source_windows]
            )
            gained_features = np.stack([extract_features(window) for window in gained])
            gain_score = _score(model, gained_features, source_truth)
            clipped = np.stack([clip_accelerometer(window) for window in gained])
            clipped_features = np.stack([extract_features(window) for window in clipped])
            clipped_score = _score(model, clipped_features, source_truth)
            target_mask = test & (table["devices"] == TARGET_DEVICE)
            real_score = _score(model, table["features"][target_mask], table["labels"][target_mask])
            results.append(
                {
                    "fold": fold["name"],
                    "source_device": source,
                    "target_device": TARGET_DEVICE,
                    "calibration_users": fold["target_calibration_users"],
                    "gain_ratio": ratio,
                    "source_macro_f1": round(source_score, 6),
                    "real_target_macro_f1": round(real_score, 6),
                    "real_degradation": round(source_score - real_score, 6),
                    "gain_synthetic_macro_f1": round(gain_score, 6),
                    "gain_synthetic_degradation": round(source_score - gain_score, 6),
                    "clipped_synthetic_macro_f1": round(clipped_score, 6),
                    "clipped_synthetic_degradation": round(source_score - clipped_score, 6),
                    **clipping_diagnostics(gained),
                }
            )
    if len(results) != 12:
        raise ValueError(f"Expected 12 one-sided cells, found {len(results)}")
    real = np.asarray([row["real_degradation"] for row in results])
    gain = np.asarray([row["gain_synthetic_degradation"] for row in results])
    clipped = np.asarray([row["clipped_synthetic_degradation"] for row in results])
    increment = np.abs(real - clipped) - np.abs(real - gain)
    payload = {
        "protocol_commit": "417f81f",
        "method": "gain followed by fixed plus-or-minus 2g clipping for s3mini targets only",
        "rail_m_s2": S3MINI_RAIL_M_S2,
        "summary": {
            "evaluations": len(results),
            **_extended_fit(real, gain, "gain"),
            **_extended_fit(real, clipped, "clipped"),
            "clipped_minus_gain_absolute_error": float(np.mean(increment)),
            "cells_improved": int(np.sum(increment < 0)),
            "cells_equal": int(np.sum(increment == 0)),
            "cells_worsened": int(np.sum(increment > 0)),
            "mean_changed_scalar_fraction": float(
                np.mean([row["changed_scalar_fraction"] for row in results])
            ),
            "mean_changed_window_fraction": float(
                np.mean([row["changed_window_fraction"] for row in results])
            ),
        },
        "profiles": list(profiles.values()),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
