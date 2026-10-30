"""Rotation-invariant accelerometer gain from low-gyroscope samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synthshift.devices.noise import GRAVITY_M_S2, GYRO_MAX_RAD_S, MIN_DIFFERENCE_PAIRS

MIN_STILL_SAMPLES = MIN_DIFFERENCE_PAIRS


def still_acceleration_magnitude(values: np.ndarray) -> np.ndarray:
    """Acceleration magnitude at samples whose gyroscope is near still.

    Activity labels are not read. The acceleration magnitude itself is not
    gated, so a gain error is not removed before it is measured.
    """

    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Expected a [time, 6] IMU array")
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)
    gyroscope = np.linalg.norm(values[:, 3:], axis=1)
    acceleration = np.linalg.norm(values[:, :3], axis=1)
    return acceleration[gyroscope <= GYRO_MAX_RAD_S].astype(np.float64)


def accelerometer_gain(magnitudes: np.ndarray) -> float:
    """Median still acceleration magnitude relative to standard gravity."""

    if magnitudes.ndim != 1 or magnitudes.size < MIN_STILL_SAMPLES:
        raise ValueError("Need enough still acceleration magnitudes to estimate gain")
    return float(np.median(magnitudes) / GRAVITY_M_S2)


def gain_ratio(source_gain: float, target_gain: float) -> float:
    if source_gain <= 0 or target_gain <= 0:
        raise ValueError("Accelerometer gain must be positive")
    return float(target_gain / source_gain)


def apply_accelerometer_gain(values: np.ndarray, ratio: float) -> np.ndarray:
    """Scale the three acceleration channels and leave the gyroscope unchanged."""

    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Expected a [time, 6] IMU array")
    transformed = np.array(values, dtype=np.float64, copy=True)
    transformed[:, :3] *= ratio
    return transformed


def estimate_gain_profile(
    manifest_path: Path,
    segment_root: Path,
    users: list[str],
    device: str,
) -> dict[str, object]:
    """Equal-user-weighted gain from calibration users, without activity labels."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    per_user_magnitudes: dict[str, list[np.ndarray]] = {user: [] for user in users}
    for record in manifest["segments"]:
        if record["user"] not in per_user_magnitudes or record["device"] != device:
            continue
        with np.load(segment_root / record["file"]) as bundle:
            magnitudes = still_acceleration_magnitude(bundle["values"])
        if len(magnitudes):
            per_user_magnitudes[record["user"]].append(magnitudes)
    per_user = {}
    usable_gains = []
    for user in users:
        parts = per_user_magnitudes[user]
        pooled = np.concatenate(parts) if parts else np.empty(0)
        enough = len(pooled) >= MIN_STILL_SAMPLES
        gain = accelerometer_gain(pooled) if enough else None
        per_user[user] = {"still_samples": int(len(pooled)), "gain": gain}
        if gain is not None:
            usable_gains.append(gain)
    if not usable_gains:
        raise ValueError(f"No usable still acceleration samples for device={device}")
    return {
        "device": device,
        "users": users,
        "usable_users": [user for user in users if per_user[user]["gain"] is not None],
        "samples_by_user": per_user,
        "gain": float(np.mean(usable_gains)),
    }
