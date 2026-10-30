"""Identity check between the HHAR still experiment and activity devices."""

from __future__ import annotations

from pathlib import Path


def index_still_devices(root: Path) -> list[dict[str, str]]:
    """List still-experiment device folders without reading sample values."""

    records = []
    for path in sorted(root.rglob("*.csv")):
        if path.name.startswith(".") or path.name == "x-sense.csv":
            continue
        relative = path.relative_to(root)
        if len(relative.parts) < 3:
            continue
        orientation, device_name, filename = relative.parts[-3:]
        records.append(
            {
                "orientation": orientation,
                "device_name": device_name,
                "model": Path(filename).stem,
            }
        )
    return records


def still_identity_report(root: Path, activity_devices: list[str]) -> dict[str, object]:
    """Decide whether still files name the same physical phones as the activity set."""

    records = index_still_devices(root)
    still_names = sorted({record["device_name"] for record in records})
    shared = sorted(set(activity_devices) & set(still_names))
    by_model: dict[str, list[str]] = {}
    for record in records:
        by_model.setdefault(record["model"], [])
        if record["device_name"] not in by_model[record["model"]]:
            by_model[record["model"]].append(record["device_name"])
    return {
        "still_device_names": still_names,
        "activity_devices": list(activity_devices),
        "shared_names": shared,
        "device_names_by_model": {model: names for model, names in sorted(by_model.items())},
        "use_for_instance_noise": False,
        "reason": (
            "Still-experiment folders use lab device names. A shared folder string "
            "does not identify every activity instance, so still files are not used "
            "to estimate instance noise."
        ),
    }
