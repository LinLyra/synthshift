"""Unlabelled per-axis gyroscope bias from low-motion IMU samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synthshift.devices.noise import (
    ACCELERATION_RADIUS_M_S2,
    GRAVITY_M_S2,
    GYRO_MAX_RAD_S,
)

MIN_LOW_MOTION_SAMPLES = 2_000


def low_motion_gyroscope(values: np.ndarray, gyro_max_rad_s: float = GYRO_MAX_RAD_S) -> np.ndarray:
    """Return gyroscope vectors inside the protocol's label-free low-motion gate."""

    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Expected a [time, 6] IMU array")
    if gyro_max_rad_s <= 0:
        raise ValueError("Gyroscope threshold must be positive")
    acceleration = np.linalg.norm(values[:, :3], axis=1)
    gyroscope = np.linalg.norm(values[:, 3:], axis=1)
    eligible = (np.abs(acceleration - GRAVITY_M_S2) <= ACCELERATION_RADIUS_M_S2) & (
        gyroscope <= gyro_max_rad_s
    )
    return values[eligible, 3:].astype(np.float64)


def gyroscope_bias(samples: np.ndarray) -> np.ndarray:
    """Componentwise median bias for one calibration user."""

    if samples.ndim != 2 or samples.shape[1] != 3:
        raise ValueError("Expected a [time, 3] gyroscope array")
    if len(samples) < MIN_LOW_MOTION_SAMPLES:
        raise ValueError("Need at least 2,000 low-motion samples to estimate bias")
    return np.median(samples, axis=0)


def apply_gyroscope_bias(values: np.ndarray, bias_delta: np.ndarray) -> np.ndarray:
    """Add a three-axis bias delta without changing acceleration."""

    delta = np.asarray(bias_delta, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Expected a [time, 6] IMU array")
    if delta.shape != (3,):
        raise ValueError("Gyroscope bias delta must contain three channels")
    transformed = np.asarray(values, dtype=np.float64).copy()
    transformed[:, 3:] += delta
    return transformed


def estimate_gyroscope_bias_profile(
    manifest_path: Path,
    segment_root: Path,
    users: list[str],
    device: str,
    *,
    gyro_max_rad_s: float = GYRO_MAX_RAD_S,
) -> dict[str, object]:
    """Estimate a profile only when every named calibration user is usable."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    per_user_parts: dict[str, list[np.ndarray]] = {user: [] for user in users}
    for record in manifest["segments"]:
        if record["user"] not in per_user_parts or record["device"] != device:
            continue
        with np.load(segment_root / record["file"]) as bundle:
            samples = low_motion_gyroscope(bundle["values"], gyro_max_rad_s)
        if len(samples):
            per_user_parts[record["user"]].append(samples)

    per_user: dict[str, dict[str, object]] = {}
    biases = []
    for user in users:
        parts = per_user_parts[user]
        pooled = np.concatenate(parts) if parts else np.empty((0, 3))
        bias = gyroscope_bias(pooled) if len(pooled) >= MIN_LOW_MOTION_SAMPLES else None
        per_user[user] = {
            "low_motion_samples": int(len(pooled)),
            "bias_rad_s": None if bias is None else bias.tolist(),
        }
        if bias is not None:
            biases.append(bias)

    if len(biases) != len(users):
        missing = [u for u in users if per_user[u]["bias_rad_s"] is None]
        raise ValueError(f"Unusable calibration users for device={device}: {missing}")
    stacked = np.stack(biases)
    return {
        "device": device,
        "users": users,
        "gyro_max_rad_s": gyro_max_rad_s,
        "samples_by_user": per_user,
        "bias_rad_s": np.mean(stacked, axis=0).tolist(),
        "between_user_abs_difference_rad_s": np.abs(stacked[0] - stacked[1]).tolist(),
    }
