#!/usr/bin/env python3
"""Safely unpack the official HHAR bundle and its nested experiment ZIPs."""

from __future__ import annotations

import argparse
from pathlib import Path

from synthshift.data.archive import safe_extract_zip


EXPECTED_NESTED_ARCHIVES = {
    "Activity recognition exp.zip",
    "Still exp.zip",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    outer_directory = args.output / "official_bundle"
    outer_members = safe_extract_zip(args.archive, outer_directory)
    nested = {path.name: path for path in outer_members if path.name in EXPECTED_NESTED_ARCHIVES}
    missing = EXPECTED_NESTED_ARCHIVES.difference(nested)
    if missing:
        raise ValueError(f"HHAR bundle is missing nested archives: {sorted(missing)}")

    for name, nested_archive in sorted(nested.items()):
        destination = args.output / nested_archive.stem.replace(" ", "_").lower()
        safe_extract_zip(nested_archive, destination)
        print(f"extracted {name} -> {destination}")


if __name__ == "__main__":
    main()
