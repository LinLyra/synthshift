#!/usr/bin/env python3
"""Build the frozen 135-job neural source-training manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthshift.train.jobs import write_job_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = write_job_manifest(args.split, args.output)
    print(json.dumps({key: payload[key] for key in ("protocol_commit", "job_count")}, indent=2))


if __name__ == "__main__":
    main()
