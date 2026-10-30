"""Create local user-device shards from the large HHAR CSV files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SHARD_COLUMNS = ["Creation_Time", "x", "y", "z", "User", "Device", "gt"]
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _safe_component(value: object) -> str:
    component = str(value)
    if not _SAFE_NAME.fullmatch(component):
        raise ValueError(f"Unsafe shard identifier: {component!r}")
    return component


def shard_csv(source: Path, output_directory: Path, *, chunksize: int = 250_000) -> list[Path]:
    """Split a raw sensor file in one pass, refusing to append to prior output."""

    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Shard directory is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    written: set[Path] = set()

    for frame in pd.read_csv(source, usecols=SHARD_COLUMNS, chunksize=chunksize):
        for (user, device), group in frame.groupby(["User", "Device"], sort=False):
            filename = f"{_safe_component(user)}__{_safe_component(device)}.csv"
            destination = output_directory / filename
            group.to_csv(destination, mode="a", header=destination not in written, index=False)
            written.add(destination)
    return sorted(written)
