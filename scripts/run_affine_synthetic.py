#!/usr/bin/env python3
"""Run the calibrated-affine fake-device feasibility experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthshift.eval.affine_synthetic import run_affine_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("segment_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = run_affine_experiment(
        args.features, args.split, args.manifest, args.segment_root, args.output
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
