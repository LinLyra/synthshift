#!/usr/bin/env python3
"""Run one resumable source-only neural training job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthshift.train.jobs import NeuralJob, run_source_job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("job_id")
    parser.add_argument("features", type=Path)
    parser.add_argument("raw_cache", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--train-row-limit", type=int)
    parser.add_argument("--validation-row-limit", type=int)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    row = next((item for item in manifest["jobs"] if item["job_id"] == args.job_id), None)
    if row is None:
        raise ValueError(f"Unknown job id: {args.job_id}")
    result = run_source_job(
        NeuralJob(**row),
        features_path=args.features,
        raw_cache_path=args.raw_cache,
        split_path=args.split,
        output_root=args.output_root,
        device_name=args.device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        train_row_limit=args.train_row_limit,
        validation_row_limit=args.validation_row_limit,
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
