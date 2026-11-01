#!/usr/bin/env python3
"""Audit whether synthetic target ranking beats unchanged source ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from synthshift.eval.performance_baseline import build_performance_baseline_audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("neural_artifact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    neural = json.loads(args.neural_artifact.read_text(encoding="utf-8"))
    cpu_path = Path(neural["cpu_artifact"])
    if _sha256(cpu_path) != neural["cpu_artifact_sha256"]:
        raise ValueError("CPU artifact no longer matches the frozen neural input")
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    payload = build_performance_baseline_audit(
        cpu["aggregate_results"], neural["aggregate_results"]
    )
    payload.update(
        {
            "neural_artifact": str(args.neural_artifact),
            "neural_artifact_sha256": _sha256(args.neural_artifact),
            "cpu_artifact": str(cpu_path),
            "cpu_artifact_sha256": neural["cpu_artifact_sha256"],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
