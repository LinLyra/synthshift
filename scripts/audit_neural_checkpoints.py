#!/usr/bin/env python3
"""Audit all formal neural checkpoints before target evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthshift.eval.neural import audit_formal_checkpoints
from synthshift.train.jobs import NeuralJob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = [NeuralJob(**row) for row in manifest["jobs"]]
    audit = audit_formal_checkpoints(jobs, args.output_root)
    print(json.dumps(audit, indent=2))
    if not audit["ready_for_target_evaluation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
