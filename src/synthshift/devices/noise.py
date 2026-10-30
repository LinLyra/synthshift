"""Unlabelled low-motion noise scales for one-sided white perturbation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GRAVITY_M_S2 = 9.80665
ACCELERATION_RADIUS_M_S2 = 0.5
GYRO_MAX_RAD_S = 0.15
MIN_DIFFERENCE_PAIRS = 2_000
MAD_TO_SIGMA = 1.4826


def low_motion_differences(values: np.ndarray) -> np.ndarray:
    """Return consecutive differences that both fall inside the still gate.

    The gate uses acceleration magnitude and gyroscope magnitude only. Activity
    labels are not read. Differences never cross a caller-supplied segment.
    """

    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Expected a [time, 6] IMU array")
    if len(values) < 2:
        return np.empty((0, 6), dtype=np.float64)
    acceleration = np.linalg.norm(values[:, :3], axis=1)
    gyroscope = np.linalg.norm(values[:, 3:], axis=1)
    still = (np.abs(acceleration - GRAVITY_M_S2) <= ACCELERATION_RADIUS_M_S2) & (
        gyroscope <= GYRO_MAX_RAD_S
    )
    consecutive = still[:-1] & still[1:]
    return np.diff(values.astype(np.float64), axis=0)[consecutive]


def difference_sigma(differences: np.ndarray) -> np.ndarray:
    """Convert robust difference scales into equivalent white-noise sample scales."""

    if differences.ndim != 2 or differences.shape[1] != 6 or len(differences) == 0:
        raise ValueError("Need at least one six-channel difference")
    center = np.median(differences, axis=0)
    mad = np.median(np.abs(differences - center), axis=0)
    return MAD_TO_SIGMA * mad / np.sqrt(2.0)


def added_noise_scale(source_sigma: np.ndarray, target_sigma: np.ndarray) -> np.ndarray:
    """White noise that can raise, but not lower, a source channel's still scale."""

    source = np.asarray(source_sigma, dtype=np.float64)
    target = np.asarray(target_sigma, dtype=np.float64)
    if source.shape != (6,) or target.shape != (6,):
        raise ValueError("Noise scales must contain six channels")
    gap = np.maximum(0.0, np.square(target) - np.square(source))
    return np.sqrt(gap)


def apply_white_noise(
    values: np.ndarray, sigma: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=np.float64)
    if values.shape[-1] != sigma.shape[0]:
        raise ValueError("Noise scale does not match sensor channels")
    if np.all(sigma == 0):
        return np.array(values, copy=True)
    return values + rng.normal(0.0, 1.0, size=values.shape) * sigma


def estimate_noise_profile(
    manifest_path: Path,
    segment_root: Path,
    users: list[str],
    device: str,
) -> dict[str, object]:
    """Estimate an equal-user-weighted still-noise scale without activity labels."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        record
        for record in manifest["segments"]
        if record["user"] in users and record["device"] == device
    ]
    per_user_diffs = {user: [] for user in users}
    for record in records:
        with np.load(segment_root / record["file"]) as bundle:
            differences = low_motion_differences(bundle["values"])
        if len(differences):
            per_user_diffs[record["user"]].append(differences)
    per_user = {}
    usable_sigmas = []
    for user in users:
        parts = per_user_diffs[user]
        pooled = np.concatenate(parts) if parts else np.empty((0, 6))
        enough = len(pooled) >= MIN_DIFFERENCE_PAIRS
        sigma = difference_sigma(pooled) if enough else None
        per_user[user] = {
            "difference_pairs": int(len(pooled)),
            "sigma": None if sigma is None else sigma.tolist(),
        }
        if sigma is not None:
            usable_sigmas.append(sigma)
    if not usable_sigmas:
        raise ValueError(f"No usable still-noise samples for device={device}")
    combined = np.mean(np.stack(usable_sigmas), axis=0)
    return {
        "device": device,
        "users": users,
        "usable_users": [user for user in users if per_user[user]["sigma"] is not None],
        "samples_by_user": per_user,
        "sigma": combined.tolist(),
    }
