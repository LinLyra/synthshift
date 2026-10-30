#!/usr/bin/env python3
"""Run the frozen four-condition gain and gyroscope-bias experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthshift.eval.gyro_bias_factor import run_gyro_bias_factor_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("segment_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gyro-max-rad-s", type=float, default=0.15)
    args = parser.parse_args()
    payload = run_gyro_bias_factor_experiment(
        args.features,
        args.split,
        args.manifest,
        args.segment_root,
        args.output,
        gyro_max_rad_s=args.gyro_max_rad_s,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
