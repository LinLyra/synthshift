#!/usr/bin/env python3
"""Build local five-second statistical features from resampled HHAR segments."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthshift.data.features import build_feature_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("segment_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(build_feature_table(args.manifest, args.segment_root, args.output))


if __name__ == "__main__":
    main()
