#!/usr/bin/env python3
"""Run a source-only reduced neural lifecycle smoke test without target evaluation."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from synthshift.models.neural import build_neural_model, parameter_count
from synthshift.train.neural import checkpoint_sha256, refit_exact_epochs, select_best_epoch


def _stratified_subset(
    indices: np.ndarray, labels: np.ndarray, limit: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels[indices])
    base, remainder = divmod(limit, len(classes))
    selected = []
    for position, label in enumerate(classes):
        candidates = indices[labels[indices] == label]
        take = min(len(candidates), base + (position < remainder))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("raw_cache", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--source", default="nexus4_1")
    parser.add_argument("--train-rows", type=int, default=384)
    parser.add_argument("--validation-rows", type=int, default=192)
    parser.add_argument("--selection-epochs", type=int, default=2)
    args = parser.parse_args()

    with np.load(args.features) as bundle:
        users = bundle["users"]
        devices = bundle["devices"]
        labels = bundle["labels"]
    raw = np.load(args.raw_cache, mmap_mode="r")
    config = json.loads(args.split.read_text(encoding="utf-8"))
    fold = next(item for item in config["folds"] if item["name"] == args.fold)
    source = devices == args.source
    train_candidates = np.flatnonzero(source & np.isin(users, fold["train_users"]))
    validation_candidates = np.flatnonzero(source & np.isin(users, fold["validation_users"]))
    train_indices = _stratified_subset(train_candidates, labels, args.train_rows, seed=100)
    validation_indices = _stratified_subset(
        validation_candidates, labels, args.validation_rows, seed=101
    )
    classes = np.unique(labels[train_indices])
    if set(classes) != set(np.unique(labels[validation_indices])):
        raise ValueError("Smoke subsets must contain the same classes")

    device = torch.device("cpu")
    architectures = []
    for architecture in ("compact_cnn", "deepconvlstm", "resnet1d"):
        started = time.perf_counter()
        selection = select_best_epoch(
            architecture,
            np.asarray(raw[train_indices]),
            labels[train_indices],
            np.asarray(raw[validation_indices]),
            labels[validation_indices],
            classes,
            seed=0,
            device=device,
            max_epochs=args.selection_epochs,
            patience=max(1, args.selection_epochs),
        )
        combined = np.concatenate([train_indices, validation_indices])
        model, normalization, refit_losses = refit_exact_epochs(
            architecture,
            np.asarray(raw[combined]),
            labels[combined],
            classes,
            seed=0,
            epochs=selection.best_epoch,
            device=device,
        )
        architectures.append(
            {
                "architecture": architecture,
                "parameters": parameter_count(build_neural_model(architecture, len(classes))),
                "best_epoch": selection.best_epoch,
                "selection_epochs_run": selection.epochs_run,
                "source_validation_macro_f1": selection.best_validation_macro_f1,
                "selection_seconds": selection.elapsed_seconds,
                "total_selection_and_refit_seconds": time.perf_counter() - started,
                "refit_losses": list(refit_losses),
                "normalization_mean": normalization.mean.tolist(),
                "normalization_std": normalization.std.tolist(),
                "checkpoint_sha256": checkpoint_sha256(model),
            }
        )

    payload = {
        "protocol_commit": "b564116",
        "training_implementation_commit": "5e1a7d7",
        "scope": (
            "source-only full-split lifecycle throughput benchmark; no target device evaluated"
            if len(train_indices) == len(train_candidates)
            and len(validation_indices) == len(validation_candidates)
            else "source-only reduced lifecycle smoke test; no target device evaluated"
        ),
        "fold": args.fold,
        "source_device": args.source,
        "train_users": fold["train_users"],
        "validation_users": fold["validation_users"],
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "selection_epoch_cap": args.selection_epochs,
        "classes": classes.tolist(),
        "backend": str(device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "maximum_resident_set_size_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "architectures": architectures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
