#!/usr/bin/env python3
"""Audit cross-phone Arrival_Time alignment on calibration users only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthshift.data.splits import load_and_validate_split
from synthshift.data.sync import build_sync_report, load_accelerometer_streams, write_sync_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("accelerometer", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-lag-ms", type=int, default=120_000)
    args = parser.parse_args()
    config = load_and_validate_split(args.split)
    users = {user for fold in config["folds"] for user in fold["target_calibration_users"]}
    devices = set(config["primary_devices"])
    streams = load_accelerometer_streams(args.accelerometer, users, devices)
    report = build_sync_report(streams, config, max_lag_ms=args.max_lag_ms)
    write_sync_report(report, args.output)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
