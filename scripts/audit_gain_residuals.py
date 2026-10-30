#!/usr/bin/env python3
"""Audit committed calibration-side factors against gain residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthshift.eval.residual_audit import build_residual_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gain", type=Path)
    parser.add_argument("noise", type=Path)
    parser.add_argument("bias", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = build_residual_audit(args.gain, args.noise, args.bias, args.output)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
