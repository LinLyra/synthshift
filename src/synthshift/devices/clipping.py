"""Fixed accelerometer clipping for the s3mini severe-device case study."""

from __future__ import annotations

import numpy as np

from synthshift.devices.noise import GRAVITY_M_S2

S3MINI_RAIL_M_S2 = 2.0 * GRAVITY_M_S2


def clip_accelerometer(values: np.ndarray, rail_m_s2: float = S3MINI_RAIL_M_S2) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Expected a [time, 6] IMU array")
    if rail_m_s2 <= 0:
        raise ValueError("Clipping rail must be positive")
    transformed = np.asarray(values, dtype=np.float64).copy()
    transformed[:, :3] = np.clip(transformed[:, :3], -rail_m_s2, rail_m_s2)
    return transformed


def clipping_diagnostics(
    windows: np.ndarray, rail_m_s2: float = S3MINI_RAIL_M_S2
) -> dict[str, float]:
    if windows.ndim != 3 or windows.shape[2] != 6:
        raise ValueError("Expected a [windows, time, 6] IMU array")
    acceleration = np.abs(windows[:, :, :3])
    changed = acceleration > rail_m_s2
    return {
        "changed_scalar_fraction": float(np.mean(changed)),
        "changed_window_fraction": float(np.mean(np.any(changed, axis=(1, 2)))),
        "maximum_absolute_before": float(np.max(acceleration)),
        "maximum_absolute_after": float(min(np.max(acceleration), rail_m_s2)),
    }
