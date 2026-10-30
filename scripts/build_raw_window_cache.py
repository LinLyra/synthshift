#!/usr/bin/env python3
"""Build and checksum the raw 5-second neural-training window cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthshift.data.raw_window_cache import build_raw_window_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("segment_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    result = build_raw_window_cache(
        args.features, args.segment_root, args.output, args.manifest
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
