#!/usr/bin/env python3
"""Shard HHAR accelerometer and gyroscope files for stream-wise processing."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthshift.data.shard import shard_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("accelerometer", type=Path)
    parser.add_argument("gyroscope", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    shard_csv(args.accelerometer, args.output / "accelerometer")
    shard_csv(args.gyroscope, args.output / "gyroscope")


if __name__ == "__main__":
    main()
