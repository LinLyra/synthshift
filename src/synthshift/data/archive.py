"""Safe archive extraction helpers for research data acquisition."""

from __future__ import annotations

import shutil
import stat
from pathlib import Path
from zipfile import ZipFile, ZipInfo


def _validated_destination(root: Path, member: ZipInfo) -> Path:
    """Resolve a ZIP member below ``root`` or reject the archive."""

    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"Refusing symbolic link in ZIP: {member.filename}")

    destination = (root / member.filename).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Refusing path traversal in ZIP: {member.filename}") from error
    return destination


def safe_extract_zip(archive: Path, output_directory: Path) -> list[Path]:
    """Extract a ZIP without allowing links or paths outside the destination."""

    extracted: list[Path] = []
    output_directory.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as bundle:
        bad_member = bundle.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP integrity failure in member: {bad_member}")

        for member in bundle.infolist():
            destination = _validated_destination(output_directory, member)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted.append(destination)
    return extracted
