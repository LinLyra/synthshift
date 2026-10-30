#!/usr/bin/env python3
"""Run the unlabelled accelerometer-gain fake-device experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthshift.eval.gain_synthetic import run_gain_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("segment_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--affine", type=Path)
    parser.add_argument("--noise", type=Path)
    args = parser.parse_args()
    comparisons = {}
    if args.affine is not None:
        comparisons["affine"] = args.affine
    if args.noise is not None:
        comparisons["noise"] = args.noise
    payload = run_gain_experiment(
        args.features,
        args.split,
        args.manifest,
        args.segment_root,
        args.output,
        comparison_paths=comparisons,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
