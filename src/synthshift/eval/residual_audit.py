"""Descriptive audit of calibration-side factors against gain residuals."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr

from synthshift.devices.noise import added_noise_scale
from synthshift.eval.noise_synthetic import _finite

CONTINUOUS_FACTORS = (
    "absolute_log_gain_ratio",
    "gyroscope_bias_delta_l2",
    "gyroscope_bias_user_disagreement_max",
    "noise_sigma_delta_l2",
    "one_sided_added_noise_l2",
)
OUTCOMES = ("signed_gain_residual", "absolute_gain_error")


def _profile_map(profiles: list[dict[str, object]]) -> dict[tuple[tuple[str, ...], str], dict]:
    mapped = {}
    for profile in profiles:
        key = (tuple(profile["users"]), str(profile["device"]))
        if key in mapped:
            raise ValueError(f"Duplicate profile key: {key}")
        mapped[key] = profile
    return mapped


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return {
            "pearson_r": _finite(pearsonr(x, y).statistic),
            "spearman_r": _finite(spearmanr(x, y).statistic),
        }


def build_residual_audit(
    gain_path: Path,
    noise_path: Path,
    bias_path: Path,
    output_path: Path,
) -> dict[str, object]:
    gain_payload = json.loads(gain_path.read_text(encoding="utf-8"))
    noise_payload = json.loads(noise_path.read_text(encoding="utf-8"))
    bias_payload = json.loads(bias_path.read_text(encoding="utf-8"))
    gains = _profile_map(gain_payload["profiles"])
    noises = _profile_map(noise_payload["profiles"])
    biases = _profile_map(bias_payload["gyroscope_bias_profiles"])
    rows = []

    for result in gain_payload["results"]:
        source = str(result["source_device"])
        target = str(result["target_device"])
        if source == target:
            continue
        users = tuple(result["calibration_users"])
        source_key, target_key = (users, source), (users, target)
        source_gain = float(gains[source_key]["gain"])
        target_gain = float(gains[target_key]["gain"])
        source_bias = np.asarray(biases[source_key]["bias_rad_s"], dtype=np.float64)
        target_bias = np.asarray(biases[target_key]["bias_rad_s"], dtype=np.float64)
        source_disagreement = np.linalg.norm(
            biases[source_key]["between_user_abs_difference_rad_s"]
        )
        target_disagreement = np.linalg.norm(
            biases[target_key]["between_user_abs_difference_rad_s"]
        )
        source_noise = np.asarray(noises[source_key]["sigma"], dtype=np.float64)
        target_noise = np.asarray(noises[target_key]["sigma"], dtype=np.float64)
        signed_residual = float(result["real_degradation"]) - float(
            result["synthetic_degradation"]
        )
        rows.append(
            {
                "fold": result["fold"],
                "source_device": source,
                "target_device": target,
                "calibration_users": list(users),
                "real_degradation": result["real_degradation"],
                "gain_synthetic_degradation": result["synthetic_degradation"],
                "signed_gain_residual": signed_residual,
                "absolute_gain_error": abs(signed_residual),
                "absolute_log_gain_ratio": abs(np.log(target_gain / source_gain)),
                "gyroscope_bias_delta_l2": float(np.linalg.norm(target_bias - source_bias)),
                "gyroscope_bias_user_disagreement_max": float(
                    max(source_disagreement, target_disagreement)
                ),
                "noise_sigma_delta_l2": float(np.linalg.norm(target_noise - source_noise)),
                "one_sided_added_noise_l2": float(
                    np.linalg.norm(added_noise_scale(source_noise, target_noise))
                ),
                "same_model": int(source.rsplit("_", 1)[0] == target.rsplit("_", 1)[0]),
            }
        )

    if len(rows) != 60:
        raise ValueError(f"Expected 60 off-diagonal cells, found {len(rows)}")
    correlations = {}
    for factor in CONTINUOUS_FACTORS:
        x = np.asarray([row[factor] for row in rows], dtype=np.float64)
        correlations[factor] = {}
        for outcome in OUTCOMES:
            y = np.asarray([row[outcome] for row in rows], dtype=np.float64)
            correlations[factor][outcome] = _correlation(x, y)

    def grouped(field: str) -> list[dict[str, object]]:
        keys = sorted({str(row[field]) for row in rows})
        summaries = []
        for key in keys:
            selected = [row for row in rows if str(row[field]) == key]
            residual = np.asarray([row["signed_gain_residual"] for row in selected])
            summaries.append(
                {
                    field: key,
                    "cells": len(selected),
                    "mean_signed_gain_residual": float(np.mean(residual)),
                    "mean_absolute_gain_error": float(np.mean(np.abs(residual))),
                }
            )
        return summaries

    pair_groups = {}
    for row in rows:
        key = "|".join(sorted((row["source_device"], row["target_device"])))
        pair_groups.setdefault(key, []).append(row)
    pair_summary = []
    for pair, selected in sorted(pair_groups.items()):
        residual = np.asarray([row["signed_gain_residual"] for row in selected])
        pair_summary.append(
            {
                "unordered_device_pair": pair,
                "cells": len(selected),
                "mean_signed_gain_residual": float(np.mean(residual)),
                "mean_absolute_gain_error": float(np.mean(np.abs(residual))),
            }
        )

    same_model = [row for row in rows if row["same_model"] == 1]
    cross_model = [row for row in rows if row["same_model"] == 0]

    def subset(selected: list[dict[str, object]]) -> dict[str, float | int]:
        residual = np.asarray([row["signed_gain_residual"] for row in selected])
        return {
            "cells": len(selected),
            "mean_signed_gain_residual": float(np.mean(residual)),
            "median_signed_gain_residual": float(np.median(residual)),
            "mean_absolute_gain_error": float(np.mean(np.abs(residual))),
        }

    ordered = sorted(rows, key=lambda row: row["signed_gain_residual"])
    payload = {
        "protocol_commit": "cc76dd4",
        "method": "descriptive calibration-only factor audit of gain residuals",
        "summary": {
            "cells": len(rows),
            "same_model": subset(same_model),
            "cross_model": subset(cross_model),
            "correlations": correlations,
            "folds": grouped("fold"),
            "unordered_device_pairs": pair_summary,
            "largest_negative_signed_residuals": ordered[:5],
            "largest_positive_signed_residuals": ordered[-5:][::-1],
        },
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
