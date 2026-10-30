#!/usr/bin/env python3
"""Run the unlabelled low-motion noise fake-device experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthshift.data.splits import load_and_validate_split
from synthshift.data.still import still_identity_report
from synthshift.eval.noise_synthetic import run_noise_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("segment_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--still-root", type=Path)
    parser.add_argument("--affine", type=Path)
    args = parser.parse_args()
    config = load_and_validate_split(args.split)
    still_report = (
        None
        if args.still_root is None
        else still_identity_report(args.still_root, list(config["primary_devices"]))
    )
    payload = run_noise_experiment(
        args.features,
        args.split,
        args.manifest,
        args.segment_root,
        args.output,
        affine_path=args.affine,
        still_report=still_report,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
