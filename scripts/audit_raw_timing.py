#!/usr/bin/env python3
"""Run the frozen raw HHAR timing audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthshift.eval.raw_timing_audit import run_raw_timing_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_root", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("residual", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = run_raw_timing_audit(args.shard_root, args.split, args.residual, args.output)
    compact = {k: v for k, v in payload["summary"].items() if k != "fold_correlations"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
