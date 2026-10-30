"""Unlabelled robust device profiles and affine calibration transforms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def estimate_device_profile(
    manifest_path: Path,
    segment_root: Path,
    users: list[str],
    device: str,
    *,
    max_samples_per_user: int = 50_000,
) -> dict[str, object]:
    """Estimate an equal-user-weighted profile without reading activity labels."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records_by_user = {
        user: [
            record
            for record in manifest["segments"]
            if record["user"] == user and record["device"] == device
        ]
        for user in users
    }
    user_samples = []
    counts = {}
    for user, records in records_by_user.items():
        chunks = []
        for record in records:
            with np.load(segment_root / record["file"]) as bundle:
                chunks.append(bundle["values"])
        if not chunks:
            raise ValueError(f"No calibration samples for user={user}, device={device}")
        values = np.concatenate(chunks)
        if len(values) > max_samples_per_user:
            indices = np.linspace(0, len(values) - 1, max_samples_per_user, dtype=np.int64)
            values = values[indices]
        user_samples.append(values)
        counts[user] = len(values)
    samples = np.concatenate(user_samples)
    q25, median, q75 = np.quantile(samples, [0.25, 0.5, 0.75], axis=0)
    return {
        "device": device,
        "users": users,
        "samples_by_user": counts,
        "median": median.tolist(),
        "iqr": (q75 - q25).tolist(),
    }


def affine_parameters(
    source_profile: dict[str, object], target_profile: dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    source_median = np.asarray(source_profile["median"], dtype=np.float64)
    target_median = np.asarray(target_profile["median"], dtype=np.float64)
    source_iqr = np.asarray(source_profile["iqr"], dtype=np.float64)
    target_iqr = np.asarray(target_profile["iqr"], dtype=np.float64)
    if np.any(source_iqr <= 1e-8) or np.any(target_iqr <= 1e-8):
        raise ValueError("Cannot calibrate a channel with near-zero IQR")
    scale = target_iqr / source_iqr
    offset = target_median - source_median * scale
    return scale, offset


def apply_affine(values: np.ndarray, scale: np.ndarray, offset: np.ndarray) -> np.ndarray:
    if values.shape[-1] != len(scale) or len(scale) != len(offset):
        raise ValueError("Affine parameters do not match sensor channels")
    return values * scale + offset
