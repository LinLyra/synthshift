"""Raw exported-value discreteness and extreme-occupancy profiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUMMARY_METRICS = (
    "unique_value_fraction_median",
    "consecutive_repeat_fraction_median",
    "unique_spacing_p10_median",
    "unique_spacing_p50_median",
    "extreme_occupancy_max",
    "within_one_spacing_extreme_occupancy_max",
    "longest_extreme_run_max",
    "robust_range_median",
    "extreme_symmetry_error_median",
)


def _longest_true_run(mask: np.ndarray) -> int:
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    lengths = edges[1::2] - edges[::2]
    return int(np.max(lengths)) if len(lengths) else 0


def channel_representation_metrics(values: np.ndarray) -> dict[str, float | int]:
    """Compute protocol-fixed metrics for one raw exported channel."""

    channel = np.asarray(values, dtype=np.float64)
    if channel.ndim != 1 or len(channel) < 2 or not np.all(np.isfinite(channel)):
        raise ValueError("Need at least two finite one-dimensional values")
    unique = np.unique(channel)
    spacing = np.diff(unique)
    positive_spacing = spacing[spacing > 0]
    if not len(positive_spacing):
        raise ValueError("Channel has no positive unique-value spacing")
    spacing_p10, spacing_p50 = np.percentile(positive_spacing, [10, 50])
    minimum, maximum = float(unique[0]), float(unique[-1])
    minimum_mask = channel == minimum
    maximum_mask = channel == maximum
    extreme_mask = minimum_mask | maximum_mask
    low_quantile, high_quantile = np.percentile(channel, [0.01, 99.99])
    rail_mean = (abs(minimum) + abs(maximum)) / 2.0
    symmetry_error = abs(abs(maximum) - abs(minimum)) / rail_mean if rail_mean else 0.0
    near_extreme = (channel <= minimum + spacing_p10) | (channel >= maximum - spacing_p10)
    return {
        "sample_count": int(len(channel)),
        "unique_value_fraction": float(len(unique) / len(channel)),
        "consecutive_repeat_fraction": float(np.mean(np.diff(channel) == 0)),
        "unique_spacing_p10": float(spacing_p10),
        "unique_spacing_p50": float(spacing_p50),
        "observed_minimum": minimum,
        "observed_maximum": maximum,
        "exact_minimum_occupancy": float(np.mean(minimum_mask)),
        "exact_maximum_occupancy": float(np.mean(maximum_mask)),
        "within_one_spacing_extreme_occupancy": float(np.mean(near_extreme)),
        "longest_exact_extreme_run": _longest_true_run(extreme_mask),
        "quantile_0p01": float(low_quantile),
        "quantile_99p99": float(high_quantile),
        "robust_range": float(high_quantile - low_quantile),
        "extreme_symmetry_error": float(symmetry_error),
    }


def sensor_representation_profile(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path, usecols=["x", "y", "z"])
    channels = {
        name: channel_representation_metrics(frame[name].to_numpy())
        for name in ("x", "y", "z")
    }

    def median(metric: str) -> float:
        return float(np.median([float(channels[name][metric]) for name in channels]))

    def maximum(metric: str) -> float:
        return float(np.max([float(channels[name][metric]) for name in channels]))

    summary = {
        "unique_value_fraction_median": median("unique_value_fraction"),
        "consecutive_repeat_fraction_median": median("consecutive_repeat_fraction"),
        "unique_spacing_p10_median": median("unique_spacing_p10"),
        "unique_spacing_p50_median": median("unique_spacing_p50"),
        "extreme_occupancy_max": max(
            maximum("exact_minimum_occupancy"), maximum("exact_maximum_occupancy")
        ),
        "within_one_spacing_extreme_occupancy_max": maximum(
            "within_one_spacing_extreme_occupancy"
        ),
        "longest_extreme_run_max": maximum("longest_exact_extreme_run"),
        "robust_range_median": median("robust_range"),
        "extreme_symmetry_error_median": median("extreme_symmetry_error"),
    }
    return {"path": str(path), "channels": channels, "summary": summary}
