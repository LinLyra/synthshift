"""Acquisition-order timing profiles for local HHAR CSV shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NANOSECONDS_PER_SECOND = 1_000_000_000.0
GAP_NS = 100_000_000
SEVERE_BACKWARD_NS = -1_000_000_000
BURST_NS = 1_000_000


def timing_metrics_from_timestamps(timestamps: np.ndarray) -> dict[str, float | int]:
    """Compute fixed timing metrics without sorting acquisition-order timestamps."""

    values = np.asarray(timestamps, dtype=np.int64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Need at least two one-dimensional timestamps")
    differences = np.diff(values)
    positive = differences[differences > 0]
    if not len(positive):
        raise ValueError("Need at least one positive timestamp interval")
    median = float(np.median(positive))
    if not 100_000 <= median <= NANOSECONDS_PER_SECOND:
        raise ValueError(f"Unexpected timestamp scale: median positive interval={median}")
    mad = float(np.median(np.abs(positive - median)))
    total_differences = len(differences)
    duration_s = float((np.max(values) - np.min(values)) / NANOSECONDS_PER_SECOND)
    gaps = int(np.sum(differences > GAP_NS))
    return {
        "sample_count": int(len(values)),
        "observed_duration_s": duration_s,
        "median_positive_interval_ms": median / 1_000_000.0,
        "effective_median_rate_hz": NANOSECONDS_PER_SECOND / median,
        "normalized_interval_mad": mad / median,
        "positive_interval_p95_ms": float(np.percentile(positive, 95) / 1_000_000.0),
        "positive_interval_p99_ms": float(np.percentile(positive, 99) / 1_000_000.0),
        "duplicate_fraction": float(np.sum(differences == 0) / total_differences),
        "reorder_fraction": float(np.sum(differences < 0) / total_differences),
        "severe_backward_fraction": float(
            np.sum(differences < SEVERE_BACKWARD_NS) / total_differences
        ),
        "gap_fraction": float(gaps / total_differences),
        "gaps_per_observed_minute": float(gaps / (duration_s / 60.0)),
        "burst_fraction": float(np.sum((differences > 0) & (differences <= BURST_NS))
                                / total_differences),
    }


def read_creation_timestamps(path: Path, *, chunksize: int = 250_000) -> np.ndarray:
    """Read one column in chunks while retaining acquisition order."""

    parts = []
    for chunk in pd.read_csv(path, usecols=["Creation_Time"], chunksize=chunksize):
        parts.append(chunk["Creation_Time"].to_numpy(dtype=np.int64, copy=True))
    if not parts:
        raise ValueError(f"Empty timing shard: {path}")
    return np.concatenate(parts)


def shard_timing_profile(path: Path) -> dict[str, float | int]:
    return timing_metrics_from_timestamps(read_creation_timestamps(path))
