"""Leakage-resistant segmentation and resampling for paired HHAR streams."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SENSOR_COLUMNS = ["Creation_Time", "x", "y", "z", "gt"]
VALID_LABELS = {"bike", "sit", "stairsdown", "stairsup", "stand", "walk"}


def split_segments(frame: pd.DataFrame, *, max_gap_ns: int = 100_000_000) -> list[pd.DataFrame]:
    """Split in acquisition order before any sorting or interpolation."""

    data = frame[SENSOR_COLUMNS].copy()
    data["Creation_Time"] = pd.to_numeric(data["Creation_Time"], errors="coerce")
    data["gt"] = data["gt"].fillna("<missing>").astype(str)
    valid_time = data["Creation_Time"].notna().to_numpy()
    timestamps = data["Creation_Time"].fillna(0).to_numpy(dtype=np.int64)
    labels = data["gt"].to_numpy(dtype=str)
    if len(data) == 0:
        return []

    delta = np.diff(timestamps)
    boundary = np.zeros(len(data), dtype=bool)
    boundary[0] = True
    boundary[1:] = (
        (delta < 0)
        | (delta > max_gap_ns)
        | (labels[1:] != labels[:-1])
        | (~valid_time[1:])
        | (~valid_time[:-1])
    )
    segment_ids = np.cumsum(boundary)
    segments: list[pd.DataFrame] = []
    for _, segment in data.groupby(segment_ids, sort=False):
        if segment["gt"].iloc[0] not in VALID_LABELS:
            continue
        segment = segment.dropna(subset=["Creation_Time", "x", "y", "z"])
        segment = segment.drop_duplicates("Creation_Time", keep="first")
        if len(segment) >= 2:
            segments.append(segment.reset_index(drop=True))
    return segments


def overlap_segments(
    accelerometer: list[pd.DataFrame], gyroscope: list[pd.DataFrame]
) -> list[tuple[pd.DataFrame, pd.DataFrame, int, int, str]]:
    """Find same-label temporal intersections between two segment lists."""

    overlaps = []
    for acc in accelerometer:
        acc_start, acc_end = int(acc.Creation_Time.iloc[0]), int(acc.Creation_Time.iloc[-1])
        label = str(acc["gt"].iloc[0])
        for gyro in gyroscope:
            if str(gyro["gt"].iloc[0]) != label:
                continue
            start = max(acc_start, int(gyro.Creation_Time.iloc[0]))
            end = min(acc_end, int(gyro.Creation_Time.iloc[-1]))
            if start < end:
                overlaps.append((acc, gyro, start, end, label))
    return overlaps


def resample_overlap(
    acc: pd.DataFrame,
    gyro: pd.DataFrame,
    start_ns: int,
    end_ns: int,
    *,
    rate_hz: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate both sensors on a deterministic shared integer clock grid."""

    step_ns = int(1_000_000_000 / rate_hz)
    grid_start = ((start_ns + step_ns - 1) // step_ns) * step_ns
    timestamps = np.arange(grid_start, end_ns + 1, step_ns, dtype=np.int64)
    channels = []
    for sensor in (acc, gyro):
        source_time = sensor["Creation_Time"].to_numpy(dtype=np.int64)
        axes = [np.interp(timestamps, source_time, sensor[axis]) for axis in ("x", "y", "z")]
        channels.extend(axes)
    return timestamps, np.stack(channels, axis=1).astype(np.float32)


def window_count(samples: int, window_samples: int, stride_samples: int) -> int:
    if samples < window_samples:
        return 0
    return 1 + (samples - window_samples) // stride_samples


def preprocess_pair(
    accelerometer_path: Path,
    gyroscope_path: Path,
    output_directory: Path,
    *,
    rate_hz: int = 50,
) -> list[dict[str, object]]:
    """Create resampled segment files and return their manifest records."""

    acc = pd.read_csv(accelerometer_path, usecols=SENSOR_COLUMNS)
    gyro = pd.read_csv(gyroscope_path, usecols=SENSOR_COLUMNS)
    acc_segments = split_segments(acc)
    gyro_segments = split_segments(gyro)
    records: list[dict[str, object]] = []
    stem = accelerometer_path.stem
    user, device = stem.split("__", maxsplit=1)
    output_directory.mkdir(parents=True, exist_ok=True)

    for index, (acc_segment, gyro_segment, start, end, label) in enumerate(
        overlap_segments(acc_segments, gyro_segments)
    ):
        timestamps, values = resample_overlap(
            acc_segment, gyro_segment, start, end, rate_hz=rate_hz
        )
        if len(timestamps) < 2:
            continue
        filename = f"{stem}__{index:04d}.npz"
        np.savez_compressed(
            output_directory / filename,
            timestamps=timestamps,
            values=values,
            label=np.array(label),
        )
        records.append(
            {
                "file": filename,
                "user": user,
                "device": device,
                "label": label,
                "samples": len(timestamps),
                "start_ns": int(timestamps[0]),
                "end_ns": int(timestamps[-1]),
                "windows_2_56s_50pct": window_count(len(timestamps), 128, 64),
                "windows_5s_50pct": window_count(len(timestamps), 250, 125),
                "windows_10s_50pct": window_count(len(timestamps), 500, 250),
            }
        )
    return records


def preprocess_directories(shard_root: Path, output_root: Path, manifest_path: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Preprocessed output is not empty: {output_root}")
    records: list[dict[str, object]] = []
    for accelerometer in sorted((shard_root / "accelerometer").glob("*.csv")):
        gyroscope = shard_root / "gyroscope" / accelerometer.name
        if gyroscope.is_file():
            records.extend(preprocess_pair(accelerometer, gyroscope, output_root))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "rate_hz": 50,
                "channels": ["ax", "ay", "az", "gx", "gy", "gz"],
                "segments": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
