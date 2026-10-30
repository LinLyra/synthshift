#!/usr/bin/env python3
"""Audit timestamp matching across local HHAR sensor shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthshift.data.alignment import audit_directories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_directories(
        args.shard_root / "accelerometer", args.shard_root / "gyroscope", args.output
    )


if __name__ == "__main__":
    main()
