"""Seed aggregation and frozen CPU-plus-neural ranking analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from synthshift.eval.cpu_multimodel import _ranking_metrics
from synthshift.eval.gyro_bias_factor import _extended_fit

SCORE_FIELDS = (
    "source_pooled_macro_f1",
    "real_target_pooled_macro_f1",
    "synthetic_target_pooled_macro_f1",
    "real_pooled_degradation",
    "synthetic_pooled_degradation",
    "source_subject_macro_f1",
    "real_target_subject_macro_f1",
    "synthetic_target_subject_macro_f1",
    "real_subject_degradation",
    "synthetic_subject_degradation",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_neural_seeds(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["fold"]),
            str(row["source_device"]),
            str(row["target_device"]),
            str(row["architecture"]),
        )
        grouped.setdefault(key, []).append(row)
    aggregate = []
    for key, group in sorted(grouped.items()):
        seeds = sorted(int(row["seed"]) for row in group)
        if seeds != [0, 1, 2]:
            raise ValueError(f"Expected seeds 0,1,2 for {key}; found {seeds}")
        item: dict[str, object] = {
            "fold": key[0],
            "source_device": key[1],
            "target_device": key[2],
            "configuration": f"neural_{key[3]}",
            "seed_count": 3,
        }
        item.update(
            {
                field: float(np.mean([float(row[field]) for row in group]))
                for field in SCORE_FIELDS
            }
        )
        aggregate.append(item)
    return aggregate


def _factor_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    result = {}
    configurations = sorted({str(row["configuration"]) for row in rows})
    for configuration in configurations:
        selected = [
            row
            for row in rows
            if row["configuration"] == configuration
            and row["source_device"] != row["target_device"]
        ]
        result[configuration] = {}
        for score in ("pooled", "subject"):
            real = np.asarray([row[f"real_{score}_degradation"] for row in selected])
            synthetic = np.asarray(
                [row[f"synthetic_{score}_degradation"] for row in selected]
            )
            result[configuration][score] = _extended_fit(
                real, synthetic, f"{configuration}_{score}"
            )
    return result


def combined_ranking_summary(
    cpu_rows: list[dict[str, object]], neural_rows: list[dict[str, object]]
) -> dict[str, object]:
    combined = [*cpu_rows, *neural_rows]
    configurations = sorted({str(row["configuration"]) for row in combined})
    if len(configurations) != 10:
        raise ValueError(f"Expected 10 combined configurations; found {len(configurations)}")
    off_diagonal = [row for row in combined if row["source_device"] != row["target_device"]]
    cell_keys = sorted(
        {(row["fold"], row["source_device"], row["target_device"]) for row in off_diagonal}
    )
    if len(cell_keys) != 60:
        raise ValueError(f"Expected 60 off-diagonal cells; found {len(cell_keys)}")
    cells = []
    for key in cell_keys:
        selected = sorted(
            [
                row
                for row in off_diagonal
                if (row["fold"], row["source_device"], row["target_device"]) == key
            ],
            key=lambda row: str(row["configuration"]),
        )
        if [row["configuration"] for row in selected] != configurations:
            raise ValueError(f"Incomplete configuration set for cell {key}")
        cell: dict[str, object] = {
            "fold": key[0],
            "source_device": key[1],
            "target_device": key[2],
        }
        for score in ("pooled", "subject"):
            real_degradation = np.asarray(
                [row[f"real_{score}_degradation"] for row in selected]
            )
            synthetic_degradation = np.asarray(
                [row[f"synthetic_{score}_degradation"] for row in selected]
            )
            real_target = np.asarray(
                [row[f"real_target_{score}_macro_f1"] for row in selected]
            )
            synthetic_target = np.asarray(
                [row[f"synthetic_target_{score}_macro_f1"] for row in selected]
            )
            cell[f"{score}_robustness"] = _ranking_metrics(
                real_degradation, synthetic_degradation, lower_better=True
            )
            cell[f"{score}_performance"] = _ranking_metrics(
                real_target, synthetic_target, lower_better=False
            )
        cells.append(cell)
    mean = {}
    for score in ("pooled", "subject"):
        for target in ("robustness", "performance"):
            key = f"{score}_{target}"
            mean[key] = {
                metric: float(np.nanmean([cell[key][metric] for cell in cells]))
                for metric in (
                    "kendall_tau_b",
                    "spearman_r",
                    "top2_overlap",
                    "pairwise_ordering_accuracy",
                )
            }
    return {
        "configuration_count": len(configurations),
        "configurations": configurations,
        "cell_count": len(cells),
        "mean": mean,
        "cells": cells,
    }


def build_final_neural_payload(
    seed_rows: list[dict[str, object]], cpu_artifact_path: Path
) -> dict[str, object]:
    neural_rows = aggregate_neural_seeds(seed_rows)
    if len(seed_rows) != 675 or len(neural_rows) != 225:
        raise ValueError(
            f"Expected 675 seed rows and 225 aggregates; found {len(seed_rows)} and "
            f"{len(neural_rows)}"
        )
    cpu_payload = json.loads(cpu_artifact_path.read_text(encoding="utf-8"))
    cpu_rows = cpu_payload["aggregate_results"]
    return {
        "protocol_commit": "b564116",
        "cpu_artifact": str(cpu_artifact_path),
        "cpu_artifact_sha256": sha256_file(cpu_artifact_path),
        "seed_result_count": len(seed_rows),
        "aggregate_result_count": len(neural_rows),
        "neural_factor_summary": _factor_summary(neural_rows),
        "combined_ranking": combined_ranking_summary(cpu_rows, neural_rows),
        "seed_results": seed_rows,
        "aggregate_results": neural_rows,
    }
