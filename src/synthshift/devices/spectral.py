"""Low-motion spectral-shape profiles for uniformly sampled IMU segments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synthshift.devices.noise import ACCELERATION_RADIUS_M_S2, GRAVITY_M_S2, GYRO_MAX_RAD_S

SAMPLE_RATE_HZ = 50.0
WINDOW_SAMPLES = 128
MIN_WINDOWS = 10
POWER_EPSILON = 1e-18
FEATURE_NAMES = (
    "power_fraction_0p5_2hz",
    "power_fraction_2_5hz",
    "power_fraction_5_10hz",
    "power_fraction_10_18hz",
    "spectral_centroid_hz",
    "log10_high_to_mid_power",
    "lag1_autocorrelation",
)


def low_motion_windows(values: np.ndarray) -> list[np.ndarray]:
    """Return non-overlapping windows wholly inside contiguous eligible runs."""

    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Expected a [time, 6] IMU array")
    acceleration = np.linalg.norm(values[:, :3], axis=1)
    gyroscope = np.linalg.norm(values[:, 3:], axis=1)
    eligible = (np.abs(acceleration - GRAVITY_M_S2) <= ACCELERATION_RADIUS_M_S2) & (
        gyroscope <= GYRO_MAX_RAD_S
    )
    edges = np.flatnonzero(np.diff(np.r_[False, eligible, False].astype(np.int8)))
    windows = []
    for start, end in zip(edges[::2], edges[1::2], strict=True):
        for offset in range(start, end - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
            windows.append(values[offset : offset + WINDOW_SAMPLES])
    return windows


def _channel_features(channel: np.ndarray) -> np.ndarray:
    demeaned = np.asarray(channel, dtype=np.float64) - np.mean(channel)
    taper = np.hanning(WINDOW_SAMPLES)
    spectrum = np.fft.rfft(demeaned * taper)
    power = np.square(np.abs(spectrum)) / (SAMPLE_RATE_HZ * np.sum(np.square(taper)))
    power[1:-1] *= 2.0
    frequencies = np.fft.rfftfreq(WINDOW_SAMPLES, d=1.0 / SAMPLE_RATE_HZ)
    retained = (frequencies >= 0.5) & (frequencies <= 18.0)
    retained_power = power[retained]
    retained_frequencies = frequencies[retained]
    total = float(np.sum(retained_power))
    if total <= POWER_EPSILON:
        raise ValueError("Window has insufficient non-DC spectral power")

    def band(lower: float, upper: float, *, include_upper: bool = False) -> float:
        if include_upper:
            selected = (frequencies >= lower) & (frequencies <= upper)
        else:
            selected = (frequencies >= lower) & (frequencies < upper)
        return float(np.sum(power[selected]))

    low = band(0.5, 2.0)
    middle = band(2.0, 5.0)
    upper_middle = band(5.0, 10.0)
    high = band(10.0, 18.0, include_upper=True)
    denominator = float(np.dot(demeaned, demeaned))
    lag1 = (
        float(np.dot(demeaned[:-1], demeaned[1:]) / denominator)
        if denominator > POWER_EPSILON
        else 0.0
    )
    return np.asarray(
        [
            low / total,
            middle / total,
            upper_middle / total,
            high / total,
            float(np.dot(retained_frequencies, retained_power) / total),
            float(np.log10((high + POWER_EPSILON) / (middle + POWER_EPSILON))),
            lag1,
        ]
    )


def spectral_window_features(window: np.ndarray, sensor: str) -> np.ndarray:
    if window.shape != (WINDOW_SAMPLES, 6):
        raise ValueError(f"Expected a [{WINDOW_SAMPLES}, 6] IMU window")
    if sensor not in {"accelerometer", "gyroscope"}:
        raise ValueError(f"Unsupported sensor: {sensor}")
    channels = window[:, :3] if sensor == "accelerometer" else window[:, 3:]
    return np.mean(np.stack([_channel_features(channels[:, index]) for index in range(3)]), axis=0)


def estimate_spectral_profile(
    manifest_path: Path,
    segment_root: Path,
    users: list[str],
    device: str,
    sensor: str,
) -> dict[str, object]:
    """Estimate equal-user spectral profiles without reading labels."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    per_user_features: dict[str, list[np.ndarray]] = {user: [] for user in users}
    for record in manifest["segments"]:
        if record["device"] != device or record["user"] not in per_user_features:
            continue
        with np.load(segment_root / record["file"]) as bundle:
            windows = low_motion_windows(bundle["values"])
        for window in windows:
            try:
                features = spectral_window_features(window, sensor)
            except ValueError:
                continue
            per_user_features[record["user"]].append(features)

    per_user = {}
    user_vectors = []
    for user in users:
        values = per_user_features[user]
        if len(values) < MIN_WINDOWS:
            raise ValueError(
                f"Insufficient low-motion windows for user={user}, device={device}, sensor={sensor}"
            )
        vector = np.median(np.stack(values), axis=0)
        per_user[user] = {
            "window_count": len(values),
            "features": dict(zip(FEATURE_NAMES, vector.tolist(), strict=True)),
        }
        user_vectors.append(vector)
    combined = np.mean(np.stack(user_vectors), axis=0)
    return {
        "device": device,
        "sensor": sensor,
        "users": users,
        "per_user": per_user,
        "equal_user_mean": dict(zip(FEATURE_NAMES, combined.tolist(), strict=True)),
        "between_user_absolute_difference": dict(
            zip(FEATURE_NAMES, np.abs(user_vectors[0] - user_vectors[1]).tolist(), strict=True)
        ),
    }
