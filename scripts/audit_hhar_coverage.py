#!/usr/bin/env python3
"""Profile HHAR phone stream coverage before exact sensor alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthshift.data.coverage import write_coverage_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("accelerometer", type=Path)
    parser.add_argument("gyroscope", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_coverage_report(args.accelerometer, args.gyroscope, args.output)


if __name__ == "__main__":
    main()
