"""Build an ordered float32 raw-window cache aligned to the feature table."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def build_raw_window_cache(
    features_path: Path,
    segment_root: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    window_samples: int = 250,
) -> dict[str, object]:
    with np.load(features_path) as table:
        segment_files = table["segment_files"]
        offsets = table["offsets"]
        rows = len(table["features"])
    if len(segment_files) != rows or len(offsets) != rows:
        raise ValueError("Feature metadata arrays are not row-aligned")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(rows, window_samples, 6)
    )
    loaded_name = None
    loaded_values = None
    for index, (filename_value, offset_value) in enumerate(
        zip(segment_files, offsets, strict=True)
    ):
        filename = str(filename_value)
        if filename != loaded_name:
            with np.load(segment_root / filename) as bundle:
                loaded_values = np.asarray(bundle["values"], dtype=np.float32)
            loaded_name = filename
        offset = int(offset_value)
        window = loaded_values[offset : offset + window_samples]
        if window.shape != (window_samples, 6):
            raise ValueError(f"Incomplete window at row {index}: {filename}@{offset}")
        cache[index] = window
    cache.flush()
    del cache
    manifest = {
        "features_path": str(features_path),
        "features_sha256": sha256_file(features_path),
        "segment_root": str(segment_root),
        "cache_path": str(output_path),
        "cache_sha256": sha256_file(output_path),
        "shape": [rows, window_samples, 6],
        "dtype": "float32",
        "row_alignment": "identical to features_path arrays",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
