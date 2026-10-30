"""Integrity and timing audit for HHAR raw CSV files.

The initial audit deliberately makes no training-ready assumptions. It inspects
device/user/activity coverage and estimates effective sampling rates from raw
timestamps so that resampling decisions can be documented rather than hidden in
a loader.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "Arrival_Time",
    "Creation_Time",
    "x",
    "y",
    "z",
    "User",
    "Model",
    "Device",
    "gt",
}

_TIME_SCALES = {
    "seconds": 1.0,
    "milliseconds": 1_000.0,
    "microseconds": 1_000_000.0,
    "nanoseconds": 1_000_000_000.0,
}


def validate_columns(columns: Iterable[str]) -> None:
    """Raise a useful error when an HHAR activity file has the wrong schema."""

    missing = sorted(REQUIRED_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"HHAR file is missing required columns: {', '.join(missing)}")


def infer_timestamp_scale(differences: np.ndarray) -> tuple[str, float] | None:
    """Infer a plausible timestamp unit and effective sample rate.

    HHAR timestamps are large integers, but downstream copies may convert their
    representation. We test conventional units and select the plausible rate
    closest to 100 Hz. The result is a diagnostic estimate, not a silent data
    transformation.
    """

    positive = np.asarray(differences, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size == 0:
        return None

    median_delta = float(np.median(positive))
    candidates: list[tuple[float, str, float]] = []
    for unit, ticks_per_second in _TIME_SCALES.items():
        rate = ticks_per_second / median_delta
        if 1.0 <= rate <= 2_000.0:
            distance = abs(math.log(rate / 100.0))
            candidates.append((distance, unit, rate))

    if not candidates:
        return None
    _, unit, rate = min(candidates)
    return unit, rate


def audit_csv(
    path: str | Path,
    *,
    chunksize: int = 250_000,
    max_differences_per_stream: int = 20_000,
) -> dict[str, object]:
    """Audit one HHAR activity CSV without loading the full file into memory."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    row_count = 0
    null_counts: Counter[str] = Counter()
    users: Counter[str] = Counter()
    models: Counter[str] = Counter()
    devices: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    model_by_device: Counter[str] = Counter()
    rows_by_user_device_label: Counter[str] = Counter()
    timestamp_differences: dict[str, list[float]] = defaultdict(list)
    previous_timestamp: dict[str, float] = {}

    for chunk_index, frame in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        if chunk_index == 0:
            validate_columns(frame.columns)

        row_count += len(frame)
        null_counts.update({column: int(count) for column, count in frame.isna().sum().items()})
        users.update(frame["User"].dropna().astype(str))
        models.update(frame["Model"].dropna().astype(str))
        devices.update(frame["Device"].dropna().astype(str))
        labels.update(frame["gt"].fillna("<missing>").astype(str))

        metadata = frame[["User", "Model", "Device", "gt"]].copy()
        metadata["gt"] = metadata["gt"].fillna("<missing>")
        model_device_counts = metadata.groupby(["Model", "Device"], dropna=False).size()
        for (model, device), count in model_device_counts.items():
            model_by_device[f"{model}|{device}"] += int(count)
        for (user, device, label), count in (
            metadata.groupby(["User", "Device", "gt"], dropna=False).size().items()
        ):
            rows_by_user_device_label[f"{user}|{device}|{label}"] += int(count)

        timing = frame[["User", "Device", "Creation_Time"]].dropna()
        for (user, device), group in timing.groupby(["User", "Device"], sort=False):
            key = f"{user}|{device}"
            values = pd.to_numeric(group["Creation_Time"], errors="coerce").dropna().to_numpy()
            if values.size == 0:
                continue

            if key in previous_timestamp:
                values = np.concatenate(([previous_timestamp[key]], values))
            previous_timestamp[key] = float(values[-1])

            remaining = max_differences_per_stream - len(timestamp_differences[key])
            if remaining > 0:
                diffs = np.diff(values)
                timestamp_differences[key].extend(diffs[:remaining].astype(float).tolist())

    timing_by_stream: dict[str, object] = {}
    rates_by_device: dict[str, list[float]] = defaultdict(list)
    intervals_by_device: Counter[str] = Counter()
    nonpositive_by_device: Counter[str] = Counter()
    for stream, differences in sorted(timestamp_differences.items()):
        values = np.asarray(differences, dtype=np.float64)
        positive = values[np.isfinite(values) & (values > 0)]
        estimate = infer_timestamp_scale(positive)
        _, device = stream.split("|", maxsplit=1)
        intervals_by_device[device] += int(values.size)
        nonpositive_by_device[device] += int(np.sum(~np.isfinite(values) | (values <= 0)))
        if estimate:
            rates_by_device[device].append(estimate[1])
        timing_by_stream[stream] = {
            "sampled_intervals": int(values.size),
            "nonpositive_intervals": int(np.sum(~np.isfinite(values) | (values <= 0))),
            "timestamp_unit_guess": estimate[0] if estimate else None,
            "effective_rate_hz_guess": round(estimate[1], 3) if estimate else None,
        }

    timing_by_device: dict[str, object] = {}
    for device in sorted(intervals_by_device):
        rates = np.asarray(rates_by_device[device], dtype=np.float64)
        timing_by_device[device] = {
            "streams_with_rate_estimate": int(rates.size),
            "sampled_intervals": intervals_by_device[device],
            "nonpositive_intervals": nonpositive_by_device[device],
            "median_effective_rate_hz_guess": round(float(np.median(rates)), 3)
            if rates.size
            else None,
            "min_effective_rate_hz_guess": round(float(np.min(rates)), 3) if rates.size else None,
            "max_effective_rate_hz_guess": round(float(np.max(rates)), 3) if rates.size else None,
        }

    return {
        "path": str(csv_path.resolve()),
        "rows": row_count,
        "users": dict(sorted(users.items())),
        "models": dict(sorted(models.items())),
        "devices": dict(sorted(devices.items())),
        "labels": dict(sorted(labels.items())),
        "model_by_device": dict(sorted(model_by_device.items())),
        "rows_by_user_device_label": dict(sorted(rows_by_user_device_label.items())),
        "null_counts": dict(sorted(null_counts.items())),
        "timing_by_device": timing_by_device,
        "timing_by_stream": timing_by_stream,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit an HHAR phone sensor CSV")
    parser.add_argument("csv", type=Path, help="Path to an HHAR activity CSV")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--chunksize", type=int, default=250_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_csv(args.csv, chunksize=args.chunksize)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
