#!/usr/bin/env python3
"""Create a compact, reviewable provenance manifest for a downloaded archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile, ZipFile


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(archive: Path, source_url: str) -> dict[str, object]:
    if not archive.is_file():
        raise FileNotFoundError(archive)

    try:
        with ZipFile(archive) as bundle:
            bad_member = bundle.testzip()
            if bad_member is not None:
                raise ValueError(f"ZIP integrity failure in member: {bad_member}")
            members = [
                {
                    "path": item.filename,
                    "compressed_bytes": item.compress_size,
                    "uncompressed_bytes": item.file_size,
                    "crc32": f"{item.CRC:08x}",
                }
                for item in bundle.infolist()
                if not item.is_dir()
            ]
    except BadZipFile as error:
        raise ValueError(f"Not a complete ZIP archive: {archive}") from error

    return {
        "dataset": "HHAR",
        "source_url": source_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_name": archive.name,
        "archive_bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "zip_integrity": "passed",
        "members": members,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.archive, args.source_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

