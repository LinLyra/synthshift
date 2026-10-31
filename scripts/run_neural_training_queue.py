#!/usr/bin/env python3
"""Run pending formal neural source-training jobs sequentially."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthshift.train.jobs import NeuralJob, completed_metadata, run_source_job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("raw_cache", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fold")
    parser.add_argument("--source")
    parser.add_argument("--architecture")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = [NeuralJob(**row) for row in payload["jobs"]]
    filters = {
        "fold": args.fold,
        "source_device": args.source,
        "architecture": args.architecture,
        "seed": args.seed,
    }
    for field, value in filters.items():
        if value is not None:
            jobs = [job for job in jobs if getattr(job, field) == value]
    pending = [
        job
        for job in jobs
        if completed_metadata(args.output_root / "formal" / job.job_id) is None
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(json.dumps({"selected": len(jobs), "pending_to_run": len(pending)}, indent=2))
    for position, job in enumerate(pending, start=1):
        print(f"[{position}/{len(pending)}] {job.job_id}", flush=True)
        result = run_source_job(
            job,
            features_path=args.features,
            raw_cache_path=args.raw_cache,
            split_path=args.split,
            output_root=args.output_root,
            device_name=args.device,
        )
        print(
            json.dumps(
                {
                    "job_id": job.job_id,
                    "selected_epoch": result["selected_epoch"],
                    "total_seconds": result["total_seconds"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
