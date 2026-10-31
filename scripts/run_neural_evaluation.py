#!/usr/bin/env python3
"""Run gated, resumable target evaluation and produce the final neural artifact."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from synthshift.eval.neural import audit_formal_checkpoints, evaluate_formal_job
from synthshift.eval.neural_aggregate import build_final_neural_payload
from synthshift.train.jobs import NeuralJob, completed_metadata


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("raw_cache", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("gain_artifact", type=Path)
    parser.add_argument("cpu_artifact", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = [NeuralJob(**row) for row in manifest["jobs"]]
    audit = audit_formal_checkpoints(jobs, args.run_root)
    if not audit["ready_for_target_evaluation"]:
        raise RuntimeError(
            f"Target evaluation blocked: complete={audit['complete']}/{audit['expected']}, "
            f"invalid={len(audit['invalid'])}"
        )

    evaluation_root = args.run_root / "evaluation"
    seed_rows = []
    for position, job in enumerate(jobs, start=1):
        training_metadata = completed_metadata(args.run_root / "formal" / job.job_id)
        if training_metadata is None:
            raise RuntimeError(f"Checkpoint disappeared after integrity gate: {job.job_id}")
        expected_checkpoint = training_metadata["checkpoint_sha256"]
        job_path = evaluation_root / f"{job.job_id}.json"
        if job_path.exists():
            rows = json.loads(job_path.read_text(encoding="utf-8"))
        else:
            print(f"[{position}/{len(jobs)}] {job.job_id}", flush=True)
            rows = evaluate_formal_job(
                job,
                features_path=args.features,
                raw_cache_path=args.raw_cache,
                split_path=args.split,
                gain_artifact_path=args.gain_artifact,
                output_root=args.run_root,
                device_name=args.device,
            )
            _atomic_json(job_path, rows)
        if len(rows) != 5 or any(
            row["fold"] != job.fold
            or row["source_device"] != job.source_device
            or row["architecture"] != job.architecture
            or row["seed"] != job.seed
            or row.get("checkpoint_sha256") != expected_checkpoint
            for row in rows
        ):
            raise ValueError(f"Invalid cached evaluation rows for {job.job_id}")
        seed_rows.extend(rows)
    payload = build_final_neural_payload(seed_rows, args.cpu_artifact)
    _atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "seed_result_count": payload["seed_result_count"],
                "aggregate_result_count": payload["aggregate_result_count"],
                "combined_ranking_cells": payload["combined_ranking"]["cell_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
