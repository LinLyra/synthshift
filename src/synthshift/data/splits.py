"""Validation and coverage audit for subject-disjoint HHAR folds."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

def load_and_validate_split(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    config = json.loads(raw)
    all_users = set(config["all_users"])
    test_occurrences: Counter[str] = Counter()
    names: set[str] = set()

    for fold in config["folds"]:
        name = fold["name"]
        if name in names:
            raise ValueError(f"Duplicate fold name: {name}")
        names.add(name)
        train = set(fold["train_users"])
        validation = set(fold["validation_users"])
        test = set(fold["test_users"])
        calibration = set(fold["target_calibration_users"])
        if train & validation or train & test or validation & test:
            raise ValueError(f"Subject leakage across partitions in {name}")
        if train | validation | test != all_users:
            raise ValueError(f"Fold {name} does not partition all users")
        if not calibration <= train:
            raise ValueError(f"Target calibration users must be training users in {name}")
        test_occurrences.update(test)

    if set(test_occurrences) != all_users or set(test_occurrences.values()) != {1}:
        raise ValueError("Each user must occur in exactly one outer test fold")
    config["sha256"] = hashlib.sha256(raw).hexdigest()
    return config


def audit_split_coverage(
    split_path: Path, resampled_manifest_path: Path, output_path: Path
) -> dict[str, object]:
    config = load_and_validate_split(split_path)
    manifest = json.loads(resampled_manifest_path.read_text(encoding="utf-8"))
    devices = set(config["primary_devices"])
    counts: Counter[tuple[str, str, str]] = Counter()
    for segment in manifest["segments"]:
        if segment["device"] in devices:
            counts[(segment["user"], segment["device"], segment["label"])] += segment[
                "windows_5s_50pct"
            ]

    folds = []
    for fold in config["folds"]:
        partitions = {}
        for partition_name in ("train_users", "validation_users", "test_users"):
            users = set(fold[partition_name])
            total = sum(count for (user, _, _), count in counts.items() if user in users)
            by_label = {
                label: sum(
                    count
                    for (user, _, observed_label), count in counts.items()
                    if user in users and observed_label == label
                )
                for label in sorted({key[2] for key in counts})
            }
            if any(value == 0 for value in by_label.values()):
                raise ValueError(f"Zero class coverage in {fold['name']} {partition_name}")
            partitions[partition_name] = {"windows": total, "windows_by_label": by_label}
        folds.append({"name": fold["name"], "partitions": partitions})

    report = {
        "split_config_sha256": config["sha256"],
        "window_definition": "5 seconds at 50 Hz, 50% overlap within legal segments",
        "folds": folds,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
