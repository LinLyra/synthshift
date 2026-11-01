"""Post-target source-performance baseline for synthetic ranking claims."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from synthshift.eval.cpu_multimodel import _ranking_metrics

RANKING_METRICS = (
    "kendall_tau_b",
    "spearman_r",
    "top2_overlap",
    "pairwise_ordering_accuracy",
)
BOOTSTRAP_SEED = 20261101
BOOTSTRAP_REPLICATES = 10_000


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        metric: float(np.nanmean([row[metric] for row in rows]))
        for metric in RANKING_METRICS
    }


def _paired_summary(
    source: list[dict[str, float]], synthetic: list[dict[str, float]]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for metric in RANKING_METRICS:
        differences = np.asarray(
            [
                new[metric] - baseline[metric]
                for baseline, new in zip(source, synthetic, strict=True)
            ]
        )
        tolerance = 1e-12
        result[metric] = {
            "mean_delta": float(np.nanmean(differences)),
            "median_delta": float(np.nanmedian(differences)),
            "wins": int(np.sum(differences > tolerance)),
            "ties": int(np.sum(np.abs(differences) <= tolerance)),
            "losses": int(np.sum(differences < -tolerance)),
        }
    return result


def _fold_uncertainty(
    cells: list[dict[str, object]], score: str
) -> dict[str, object]:
    folds = sorted({str(cell["fold"]) for cell in cells})
    if len(folds) != 3:
        raise ValueError(f"Expected three participant folds; found {len(folds)}")
    fold_deltas = {}
    for fold in folds:
        selected = [cell for cell in cells if cell["fold"] == fold]
        fold_deltas[fold] = {
            metric: float(
                np.nanmean(
                    [
                        cell[score]["synthetic_target"][metric]
                        - cell[score]["source_baseline"][metric]
                        for cell in selected
                    ]
                )
            )
            for metric in RANKING_METRICS
        }

    leave_one_fold_out = {}
    for excluded in folds:
        retained = [fold for fold in folds if fold != excluded]
        leave_one_fold_out[excluded] = {
            metric: float(np.mean([fold_deltas[fold][metric] for fold in retained]))
            for metric in RANKING_METRICS
        }

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = generator.integers(0, len(folds), size=(BOOTSTRAP_REPLICATES, len(folds)))
    bootstrap = {}
    for metric in RANKING_METRICS:
        values = np.asarray([fold_deltas[fold][metric] for fold in folds])
        estimates = values[draws].mean(axis=1)
        bootstrap[metric] = {
            "estimate": float(values.mean()),
            "percentile_95_interval": [
                float(value) for value in np.percentile(estimates, [2.5, 97.5])
            ],
        }
    return {
        "fold_deltas": fold_deltas,
        "leave_one_fold_out": leave_one_fold_out,
        "fold_cluster_bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "cluster_count": len(folds),
            "warning": "Only three fold clusters; intervals are descriptive and discrete.",
            "metrics": bootstrap,
        },
    }


def build_performance_baseline_audit(
    cpu_rows: Iterable[dict[str, object]], neural_rows: Iterable[dict[str, object]]
) -> dict[str, object]:
    """Compare synthetic target ranking with the unchanged source ranking."""
    combined = [*cpu_rows, *neural_rows]
    configurations = sorted({str(row["configuration"]) for row in combined})
    if len(configurations) != 10:
        raise ValueError(f"Expected 10 configurations; found {len(configurations)}")
    off_diagonal = [
        row for row in combined if row["source_device"] != row["target_device"]
    ]
    cell_keys = sorted(
        {
            (str(row["fold"]), str(row["source_device"]), str(row["target_device"]))
            for row in off_diagonal
        }
    )
    if len(cell_keys) != 60:
        raise ValueError(f"Expected 60 off-diagonal cells; found {len(cell_keys)}")

    cells = []
    for fold, source_device, target_device in cell_keys:
        selected = sorted(
            [
                row
                for row in off_diagonal
                if (
                    str(row["fold"]),
                    str(row["source_device"]),
                    str(row["target_device"]),
                )
                == (fold, source_device, target_device)
            ],
            key=lambda row: str(row["configuration"]),
        )
        if [str(row["configuration"]) for row in selected] != configurations:
            raise ValueError(
                f"Incomplete configuration set for {(fold, source_device, target_device)}"
            )
        cell: dict[str, object] = {
            "fold": fold,
            "source_device": source_device,
            "target_device": target_device,
        }
        for score in ("pooled", "subject"):
            real_target = np.asarray(
                [row[f"real_target_{score}_macro_f1"] for row in selected]
            )
            source_performance = np.asarray(
                [row[f"source_{score}_macro_f1"] for row in selected]
            )
            synthetic_target = np.asarray(
                [row[f"synthetic_target_{score}_macro_f1"] for row in selected]
            )
            cell[score] = {
                "source_baseline": _ranking_metrics(
                    real_target, source_performance, lower_better=False
                ),
                "synthetic_target": _ranking_metrics(
                    real_target, synthetic_target, lower_better=False
                ),
            }
        cells.append(cell)

    summary = {}
    for score in ("pooled", "subject"):
        source = [cell[score]["source_baseline"] for cell in cells]
        synthetic = [cell[score]["synthetic_target"] for cell in cells]
        by_fold = {}
        for fold in sorted({str(cell["fold"]) for cell in cells}):
            selected = [cell for cell in cells if cell["fold"] == fold]
            fold_source = [cell[score]["source_baseline"] for cell in selected]
            fold_synthetic = [cell[score]["synthetic_target"] for cell in selected]
            by_fold[fold] = {
                "source_baseline": _mean_metrics(fold_source),
                "synthetic_target": _mean_metrics(fold_synthetic),
                "paired_delta": {
                    metric: float(
                        np.nanmean(
                            [
                                new[metric] - baseline[metric]
                                for baseline, new in zip(
                                    fold_source, fold_synthetic, strict=True
                                )
                            ]
                        )
                    )
                    for metric in RANKING_METRICS
                },
            }
        summary[score] = {
            "source_baseline": _mean_metrics(source),
            "synthetic_target": _mean_metrics(synthetic),
            "paired_comparison": _paired_summary(source, synthetic),
            "by_fold": by_fold,
            "fold_uncertainty": _fold_uncertainty(cells, score),
        }
    return {
        "analysis_status": "post_target_diagnostic",
        "configuration_count": len(configurations),
        "configurations": configurations,
        "cell_count": len(cells),
        "summary": summary,
        "cells": cells,
    }
