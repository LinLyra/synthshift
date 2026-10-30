#!/usr/bin/env python3
"""Validate HHAR subject folds and record their class coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthshift.data.splits import audit_split_coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_split_coverage(args.split, args.manifest, args.output)


if __name__ == "__main__":
    main()
