"""Frozen calibration-user HHAR raw timing audit."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr

from synthshift.data.raw_timing import shard_timing_profile
from synthshift.eval.noise_synthetic import _finite

PROFILE_METRICS = (
    "sample_count",
    "observed_duration_s",
    "median_positive_interval_ms",
    "effective_median_rate_hz",
    "normalized_interval_mad",
    "positive_interval_p95_ms",
    "positive_interval_p99_ms",
    "duplicate_fraction",
    "reorder_fraction",
    "severe_backward_fraction",
    "gap_fraction",
    "gaps_per_observed_minute",
    "burst_fraction",
)
DIRECTED_METRICS = (
    "median_positive_interval_ms",
    "normalized_interval_mad",
    "gaps_per_observed_minute",
)


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return {
            "pearson_r": _finite(pearsonr(x, y).statistic),
            "spearman_r": _finite(spearmanr(x, y).statistic),
        }


def _equal_user_profile(per_user: dict[str, dict[str, float | int]]) -> dict[str, float]:
    if len(per_user) != 2:
        raise ValueError("A fold/device profile requires exactly two calibration users")
    return {
        metric: float(np.mean([float(profile[metric]) for profile in per_user.values()]))
        for metric in PROFILE_METRICS
    }


def run_raw_timing_audit(
    shard_root: Path,
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
                per_user = {}
                for user in fold["target_calibration_users"]:
                    path = shard_root / sensor / f"{user}__{device}.csv"
                    per_user[user] = shard_timing_profile(path)
                profiles[(fold["name"], device, sensor)] = {
                    "fold": fold["name"],
                    "device": device,
                    "sensor": sensor,
                    "users": fold["target_calibration_users"],
                    "per_user": per_user,
                    "equal_user_mean": _equal_user_profile(per_user),
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
            for metric in PROFILE_METRICS:
                source_value = float(source_profile[metric])
                target_value = float(target_profile[metric])
                row[f"{sensor}_{metric}_absolute_difference"] = abs(
                    target_value - source_value
                )
                if metric in DIRECTED_METRICS:
                    row[f"{sensor}_{metric}_target_minus_source"] = (
                        target_value - source_value
                    )
        rows.append(row)
    if len(rows) != 60:
        raise ValueError(f"Expected 60 residual cells, found {len(rows)}")

    factor_names = [
        name
        for name in rows[0]
        if name.endswith("_absolute_difference") or name.endswith("_target_minus_source")
    ]
    correlations = {}
    fold_correlations = {}
    for factor in factor_names:
        x = np.asarray([row[factor] for row in rows], dtype=np.float64)
        correlations[factor] = {
            outcome: _correlation(
                x, np.asarray([row[outcome] for row in rows], dtype=np.float64)
            )
            for outcome in ("signed_gain_residual", "absolute_gain_error")
        }
        fold_correlations[factor] = {}
        for fold in sorted({row["fold"] for row in rows}):
            selected = [row for row in rows if row["fold"] == fold]
            fold_correlations[factor][fold] = {
                outcome: _correlation(
                    np.asarray([row[factor] for row in selected], dtype=np.float64),
                    np.asarray([row[outcome] for row in selected], dtype=np.float64),
                )
                for outcome in ("signed_gain_residual", "absolute_gain_error")
            }

    payload = {
        "protocol_commit": "8cb45b5",
        "method": "acquisition-order Creation_Time audit using calibration users only",
        "summary": {
            "profile_count": len(profiles),
            "off_diagonal_cells": len(rows),
            "factor_count": len(factor_names),
            "correlations": correlations,
            "fold_correlations": fold_correlations,
        },
        "profiles": list(profiles.values()),
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
