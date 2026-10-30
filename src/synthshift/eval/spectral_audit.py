"""Frozen low-motion spectral-shape audit against gain residuals."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr

from synthshift.devices.spectral import FEATURE_NAMES, estimate_spectral_profile
from synthshift.eval.noise_synthetic import _finite


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return {
            "pearson_r": _finite(pearsonr(x, y).statistic),
            "spearman_r": _finite(spearmanr(x, y).statistic),
        }


def run_spectral_audit(
    manifest_path: Path,
    segment_root: Path,
    split_path: Path,
    residual_path: Path,
    output_path: Path,
) -> dict[str, object]:
    config = json.loads(split_path.read_text(encoding="utf-8"))
    residual_payload = json.loads(residual_path.read_text(encoding="utf-8"))
    profiles: dict[tuple[str, str, str], dict[str, object]] = {}
    for fold in config["folds"]:
        for device in config["primary_devices"]:
            for sensor in ("accelerometer", "gyroscope"):
                key = (fold["name"], device, sensor)
                profiles[key] = {
                    "fold": fold["name"],
                    **estimate_spectral_profile(
                        manifest_path,
                        segment_root,
                        fold["target_calibration_users"],
                        device,
                        sensor,
                    ),
                }

    rows = []
    for residual in residual_payload["rows"]:
        fold = str(residual["fold"])
        source = str(residual["source_device"])
        target = str(residual["target_device"])
        row = {
            "fold": fold,
            "source_device": source,
            "target_device": target,
            "signed_gain_residual": residual["signed_gain_residual"],
            "absolute_gain_error": residual["absolute_gain_error"],
        }
        for sensor in ("accelerometer", "gyroscope"):
            source_profile = profiles[(fold, source, sensor)]["equal_user_mean"]
            target_profile = profiles[(fold, target, sensor)]["equal_user_mean"]
            for feature in FEATURE_NAMES:
                source_value = float(source_profile[feature])
                target_value = float(target_profile[feature])
                row[f"{sensor}_{feature}_absolute_difference"] = abs(
                    target_value - source_value
                )
                row[f"{sensor}_{feature}_target_minus_source"] = target_value - source_value
        rows.append(row)
    if len(rows) != 60:
        raise ValueError(f"Expected 60 residual cells, found {len(rows)}")

    factors = [name for name in rows[0] if name.endswith(("_difference", "_target_minus_source"))]
    correlations = {}
    fold_correlations = {}
    for factor in factors:
        correlations[factor] = {
            outcome: _correlation(
                np.asarray([row[factor] for row in rows]),
                np.asarray([row[outcome] for row in rows]),
            )
            for outcome in ("signed_gain_residual", "absolute_gain_error")
        }
        fold_correlations[factor] = {}
        for fold in sorted({row["fold"] for row in rows}):
            selected = [row for row in rows if row["fold"] == fold]
            fold_correlations[factor][fold] = {
                outcome: _correlation(
                    np.asarray([row[factor] for row in selected]),
                    np.asarray([row[outcome] for row in selected]),
                )
                for outcome in ("signed_gain_residual", "absolute_gain_error")
            }
    payload = {
        "protocol_commit": "25da760",
        "method": "low-motion 128-sample Hann spectral-shape audit",
        "summary": {
            "profile_count": len(profiles),
            "off_diagonal_cells": len(rows),
            "factor_count": len(factors),
            "correlations": correlations,
            "fold_correlations": fold_correlations,
        },
        "profiles": list(profiles.values()),
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
