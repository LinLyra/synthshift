"""Timestamp-alignment diagnostics for paired HHAR sensor shards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_timestamps(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load, sort, and deterministically deduplicate timestamps and labels."""

    frame = pd.read_csv(path, usecols=["Creation_Time", "gt"])
    frame["Creation_Time"] = pd.to_numeric(frame["Creation_Time"], errors="coerce")
    invalid = int(frame["Creation_Time"].isna().sum())
    frame = frame.dropna(subset=["Creation_Time"])
    raw = frame["Creation_Time"].to_numpy(dtype=np.int64)
    reversals = int(np.sum(np.diff(raw) < 0)) if raw.size > 1 else 0
    frame["gt"] = frame["gt"].fillna("<missing>").astype(str)
    frame = frame.sort_values("Creation_Time", kind="stable")
    duplicates = int(frame.duplicated("Creation_Time").sum())
    frame = frame.drop_duplicates("Creation_Time", keep="first")
    return (
        frame["Creation_Time"].to_numpy(dtype=np.int64),
        frame["gt"].to_numpy(dtype=str),
        {"invalid_timestamps": invalid, "reversals": reversals, "duplicate_timestamps": duplicates},
    )


def nearest_indices(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest reference index and absolute distance for each query timestamp."""

    positions = np.searchsorted(reference, query)
    right = np.clip(positions, 0, len(reference) - 1)
    left = np.clip(positions - 1, 0, len(reference) - 1)
    left_distance = np.abs(query - reference[left])
    right_distance = np.abs(reference[right] - query)
    choose_right = right_distance < left_distance
    indices = np.where(choose_right, right, left)
    distances = np.where(choose_right, right_distance, left_distance)
    return indices, distances


def direction_stats(
    reference_time: np.ndarray,
    reference_label: np.ndarray,
    query_time: np.ndarray,
    query_label: np.ndarray,
    tolerances_ms: tuple[float, ...],
) -> dict[str, object]:
    """Measure nearest-neighbour coverage inside the shared clock range."""

    start = max(int(reference_time[0]), int(query_time[0]))
    end = min(int(reference_time[-1]), int(query_time[-1]))
    in_overlap = (query_time >= start) & (query_time <= end)
    query_time = query_time[in_overlap]
    query_label = query_label[in_overlap]
    if query_time.size == 0:
        return {
            "overlap_queries": 0,
            "nearest_distance_ms_quantiles": None,
            "matches_by_tolerance_ms": {
                str(tolerance): {
                    "coverage": 0.0,
                    "labelled_pairs": 0,
                    "label_agreement": None,
                }
                for tolerance in tolerances_ms
            },
        }
    indices, distances = nearest_indices(reference_time, query_time)
    labelled = (query_label != "<missing>") & (reference_label[indices] != "<missing>")
    label_agreement = query_label == reference_label[indices]

    matches: dict[str, object] = {}
    for tolerance_ms in tolerances_ms:
        within = distances <= tolerance_ms * 1e6
        eligible = within & labelled
        matches[str(tolerance_ms)] = {
            "coverage": round(float(np.mean(within)), 6),
            "labelled_pairs": int(np.sum(eligible)),
            "label_agreement": round(float(np.mean(label_agreement[eligible])), 6)
            if np.any(eligible)
            else None,
        }
    return {
        "overlap_queries": int(query_time.size),
        "nearest_distance_ms_quantiles": {
            str(q): round(float(np.quantile(distances, q) / 1e6), 6)
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "matches_by_tolerance_ms": matches,
    }


def audit_pair(
    accelerometer_path: Path,
    gyroscope_path: Path,
    tolerances_ms: tuple[float, ...] = (2.5, 5.0, 10.0, 20.0, 40.0),
) -> dict[str, object]:
    acc_time, acc_label, acc_cleaning = load_timestamps(accelerometer_path)
    gyro_time, gyro_label, gyro_cleaning = load_timestamps(gyroscope_path)
    if acc_time.size == 0 or gyro_time.size == 0:
        raise ValueError("Cannot align an empty sensor stream")
    return {
        "accelerometer_file": accelerometer_path.name,
        "gyroscope_file": gyroscope_path.name,
        "accelerometer_unique_timestamps": int(acc_time.size),
        "gyroscope_unique_timestamps": int(gyro_time.size),
        "accelerometer_cleaning": acc_cleaning,
        "gyroscope_cleaning": gyro_cleaning,
        "accelerometer_to_gyroscope": direction_stats(
            gyro_time, gyro_label, acc_time, acc_label, tolerances_ms
        ),
        "gyroscope_to_accelerometer": direction_stats(
            acc_time, acc_label, gyro_time, gyro_label, tolerances_ms
        ),
    }


def audit_directories(accelerometer: Path, gyroscope: Path, output: Path) -> None:
    reports = []
    for accelerometer_path in sorted(accelerometer.glob("*.csv")):
        gyroscope_path = gyroscope / accelerometer_path.name
        if gyroscope_path.is_file():
            reports.append(audit_pair(accelerometer_path, gyroscope_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"streams": reports}, indent=2) + "\n", encoding="utf-8")
