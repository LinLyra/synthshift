#!/usr/bin/env python3
"""Build segmented and resampled HHAR six-axis streams."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthshift.data.preprocess import preprocess_directories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    preprocess_directories(args.shard_root, args.output_root, args.manifest)


if __name__ == "__main__":
    main()
