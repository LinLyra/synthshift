"""Deterministic statistical features for CPU HAR baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _channel_features(window: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(window, [0.25, 0.5, 0.75], axis=0)
    return np.concatenate(
        [
            np.mean(window, axis=0),
            np.std(window, axis=0),
            np.min(window, axis=0),
            np.max(window, axis=0),
            quantiles[0],
            quantiles[1],
            quantiles[2],
            np.sqrt(np.mean(np.square(window), axis=0)),
            np.mean(np.abs(np.diff(window, axis=0)), axis=0),
        ]
    )


def extract_features(window: np.ndarray) -> np.ndarray:
    """Extract axis, magnitude, and within-sensor correlation features."""

    if window.ndim != 2 or window.shape[1] != 6:
        raise ValueError("Expected a [time, 6] IMU window")
    magnitudes = np.column_stack(
        [np.linalg.norm(window[:, :3], axis=1), np.linalg.norm(window[:, 3:], axis=1)]
    )
    correlations = []
    for sensor in (window[:, :3], window[:, 3:]):
        matrix = np.corrcoef(sensor, rowvar=False)
        correlations.extend(np.nan_to_num(matrix[np.triu_indices(3, k=1)]).tolist())
    return np.concatenate([_channel_features(window), _channel_features(magnitudes), correlations])


def build_feature_table(
    manifest_path: Path,
    segment_root: Path,
    output_path: Path,
    *,
    window_samples: int = 250,
    stride_samples: int = 125,
) -> dict[str, int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = []
    users = []
    devices = []
    labels = []
    segment_files = []
    offsets = []

    for record in manifest["segments"]:
        with np.load(segment_root / record["file"]) as bundle:
            values = bundle["values"]
        for offset in range(0, len(values) - window_samples + 1, stride_samples):
            features.append(extract_features(values[offset : offset + window_samples]))
            users.append(record["user"])
            devices.append(record["device"])
            labels.append(record["label"])
            segment_files.append(record["file"])
            offsets.append(offset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=np.asarray(features, dtype=np.float32),
        users=np.asarray(users),
        devices=np.asarray(devices),
        labels=np.asarray(labels),
        segment_files=np.asarray(segment_files),
        offsets=np.asarray(offsets, dtype=np.int32),
    )
    return {"windows": len(features), "features": int(features[0].size)}
