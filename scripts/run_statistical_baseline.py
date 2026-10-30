#!/usr/bin/env python3
"""Run the first subject-disjoint cross-device HHAR baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthshift.models.statistical_baseline import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = run_baseline(args.features, args.split, args.output)
    print(f"completed {len(results)} source-target evaluations")


if __name__ == "__main__":
    main()
