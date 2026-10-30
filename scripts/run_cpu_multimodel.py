#!/usr/bin/env python3
"""Run the frozen CPU multi-model gain and ranking audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthshift.eval.cpu_multimodel import run_cpu_multimodel_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("segment_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = run_cpu_multimodel_audit(
        args.features, args.split, args.manifest, args.segment_root, args.output
    )
    compact = {k: v for k, v in payload["summary"].items() if k != "ranking_cells"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
