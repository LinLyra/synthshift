"""Stream-level HHAR coverage profiling before sensor alignment."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def profile_csv(
    path: Path,
    *,
    chunksize: int = 250_000,
    ticks_per_second: float = 1e9,
    max_contiguous_gap_seconds: float = 0.5,
) -> dict[str, object]:
    """Estimate usable duration without bridging resets or long recording gaps."""

    previous: dict[str, tuple[float, str]] = {}
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    rows: Counter[str] = Counter()
    nonpositive: Counter[str] = Counter()
    large_gaps: Counter[str] = Counter()
    labelled_ticks: dict[str, Counter[str]] = defaultdict(Counter)
    gap_limit = max_contiguous_gap_seconds * ticks_per_second

    columns = ["Creation_Time", "User", "Device", "gt"]
    for frame in pd.read_csv(path, usecols=columns, chunksize=chunksize):
        frame["gt"] = frame["gt"].fillna("<missing>").astype(str)
        for row in frame.itertuples(index=False):
            stream = f"{row.User}|{row.Device}"
            timestamp = float(row.Creation_Time)
            label = row.gt
            rows[stream] += 1
            starts[stream] = min(starts.get(stream, timestamp), timestamp)
            ends[stream] = max(ends.get(stream, timestamp), timestamp)
            if stream in previous:
                prior_timestamp, prior_label = previous[stream]
                delta = timestamp - prior_timestamp
                if delta <= 0:
                    nonpositive[stream] += 1
                elif delta > gap_limit:
                    large_gaps[stream] += 1
                else:
                    labelled_ticks[stream][prior_label] += delta
            previous[stream] = (timestamp, label)

    streams: dict[str, object] = {}
    for stream in sorted(rows):
        streams[stream] = {
            "rows": rows[stream],
            "start_creation_time": int(starts[stream]),
            "end_creation_time": int(ends[stream]),
            "nonpositive_intervals": nonpositive[stream],
            "large_gaps": large_gaps[stream],
            "duration_seconds_by_label": {
                label: round(ticks / ticks_per_second, 3)
                for label, ticks in sorted(labelled_ticks[stream].items())
            },
        }
    return {"path": str(path.resolve()), "streams": streams}


def pair_profiles(
    accelerometer: dict[str, object], gyroscope: dict[str, object]
) -> dict[str, object]:
    """Report common streams and a conservative pre-alignment duration bound."""

    acc_streams = accelerometer["streams"]
    gyro_streams = gyroscope["streams"]
    assert isinstance(acc_streams, dict) and isinstance(gyro_streams, dict)
    paired: dict[str, object] = {}
    for stream in sorted(acc_streams.keys() & gyro_streams.keys()):
        acc = acc_streams[stream]
        gyro = gyro_streams[stream]
        assert isinstance(acc, dict) and isinstance(gyro, dict)
        acc_durations = acc["duration_seconds_by_label"]
        gyro_durations = gyro["duration_seconds_by_label"]
        assert isinstance(acc_durations, dict) and isinstance(gyro_durations, dict)
        labels = acc_durations.keys() & gyro_durations.keys()
        paired[stream] = {
            "duration_upper_bound_seconds_by_label": {
                label: min(acc_durations[label], gyro_durations[label]) for label in sorted(labels)
            }
        }
    return {
        "paired_stream_count": len(paired),
        "accelerometer_only_streams": sorted(acc_streams.keys() - gyro_streams.keys()),
        "gyroscope_only_streams": sorted(gyro_streams.keys() - acc_streams.keys()),
        "paired_streams": paired,
    }


def write_coverage_report(accelerometer_path: Path, gyroscope_path: Path, output: Path) -> None:
    accelerometer = profile_csv(accelerometer_path)
    gyroscope = profile_csv(gyroscope_path)
    report = {
        "accelerometer": accelerometer,
        "gyroscope": gyroscope,
        "pairing": pair_profiles(accelerometer, gyroscope),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
