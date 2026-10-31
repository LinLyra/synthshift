"""Resumable source-only jobs for the frozen raw-sequence neural protocol."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from synthshift.models.neural import parameter_count
from synthshift.train.neural import refit_exact_epochs, select_best_epoch

ARCHITECTURES = ("compact_cnn", "deepconvlstm", "resnet1d")
SEEDS = (0, 1, 2)
PROTOCOL_COMMIT = "b564116"


@dataclass(frozen=True)
class NeuralJob:
    job_id: str
    fold: str
    source_device: str
    architecture: str
    seed: int


def build_jobs(split_config: dict[str, object]) -> list[NeuralJob]:
    jobs = []
    for fold in split_config["folds"]:
        for source in split_config["primary_devices"]:
            for architecture in ARCHITECTURES:
                for seed in SEEDS:
                    job_id = f"{fold['name']}__{source}__{architecture}__seed{seed}"
                    jobs.append(NeuralJob(job_id, fold["name"], source, architecture, seed))
    return jobs


def write_job_manifest(split_path: Path, output_path: Path) -> dict[str, object]:
    config = json.loads(split_path.read_text(encoding="utf-8"))
    jobs = build_jobs(config)
    payload = {
        "protocol_commit": PROTOCOL_COMMIT,
        "job_count": len(jobs),
        "architectures": list(ARCHITECTURES),
        "seeds": list(SEEDS),
        "jobs": [asdict(job) for job in jobs],
    }
    _atomic_json(output_path, payload)
    return payload


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _stratified_limit(
    indices: np.ndarray, labels: np.ndarray, limit: int | None, seed: int
) -> np.ndarray:
    if limit is None or limit >= len(indices):
        return indices
    classes = np.unique(labels[indices])
    if limit < len(classes):
        raise ValueError("Row limit must retain at least one row per class")
    generator = np.random.default_rng(seed)
    base, remainder = divmod(limit, len(classes))
    selected = []
    for position, label in enumerate(classes):
        candidates = indices[labels[indices] == label]
        take = min(len(candidates), base + int(position < remainder))
        selected.extend(generator.choice(candidates, size=take, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def completed_metadata(run_directory: Path) -> dict[str, object] | None:
    metadata_path = run_directory / "metadata.json"
    checkpoint_path = run_directory / "checkpoint.pt"
    if not metadata_path.exists() or not checkpoint_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        return None
    if metadata.get("checkpoint_sha256") != _sha256(checkpoint_path):
        raise ValueError(f"Checkpoint checksum mismatch: {run_directory}")
    return metadata


def run_source_job(
    job: NeuralJob,
    *,
    features_path: Path,
    raw_cache_path: Path,
    split_path: Path,
    output_root: Path,
    device_name: str = "auto",
    max_epochs: int = 50,
    patience: int = 8,
    train_row_limit: int | None = None,
    validation_row_limit: int | None = None,
    smoke: bool = False,
) -> dict[str, object]:
    if not smoke and (
        max_epochs != 50
        or patience != 8
        or train_row_limit is not None
        or validation_row_limit is not None
    ):
        raise ValueError("Formal jobs must use the frozen optimization and complete source splits")
    run_directory = output_root / ("smoke" if smoke else "formal") / job.job_id
    existing = completed_metadata(run_directory)
    if existing is not None:
        return {**existing, "resume_action": "skipped_complete"}

    config = json.loads(split_path.read_text(encoding="utf-8"))
    fold = next(item for item in config["folds"] if item["name"] == job.fold)
    with np.load(features_path) as bundle:
        users = bundle["users"]
        devices = bundle["devices"]
        labels = bundle["labels"]
    raw = np.load(raw_cache_path, mmap_mode="r")
    if len(raw) != len(labels):
        raise ValueError("Raw cache and feature metadata are not aligned")
    source = devices == job.source_device
    train_indices = np.flatnonzero(source & np.isin(users, fold["train_users"]))
    validation_indices = np.flatnonzero(source & np.isin(users, fold["validation_users"]))
    train_indices = _stratified_limit(train_indices, labels, train_row_limit, job.seed + 1000)
    validation_indices = _stratified_limit(
        validation_indices, labels, validation_row_limit, job.seed + 2000
    )
    classes = np.unique(labels[train_indices])
    if set(classes) != set(np.unique(labels[validation_indices])):
        raise ValueError("Training and validation subsets do not contain identical classes")
    device = resolve_device(device_name)
    started = time.perf_counter()
    selection = select_best_epoch(
        job.architecture,
        np.asarray(raw[train_indices]),
        labels[train_indices],
        np.asarray(raw[validation_indices]),
        labels[validation_indices],
        classes,
        seed=job.seed,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
    )
    combined = np.concatenate([train_indices, validation_indices])
    refit_started = time.perf_counter()
    model, normalization, losses = refit_exact_epochs(
        job.architecture,
        np.asarray(raw[combined]),
        labels[combined],
        classes,
        seed=job.seed,
        epochs=selection.best_epoch,
        device=device,
    )
    refit_seconds = time.perf_counter() - refit_started
    checkpoint_path = run_directory / "checkpoint.pt"
    _atomic_checkpoint(
        checkpoint_path,
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "normalization_mean": normalization.mean,
            "normalization_std": normalization.std,
            "classes": classes,
            "job": asdict(job),
            "protocol_commit": PROTOCOL_COMMIT,
        },
    )
    metadata = {
        "status": "complete",
        "scope": "source-only training; no target device evaluated",
        "smoke": smoke,
        "protocol_commit": PROTOCOL_COMMIT,
        "job": asdict(job),
        "features_sha256": _sha256(features_path),
        "raw_cache_sha256": _sha256(raw_cache_path),
        "train_users": fold["train_users"],
        "validation_users": fold["validation_users"],
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "classes": classes.tolist(),
        "selected_epoch": selection.best_epoch,
        "selection_epochs_run": selection.epochs_run,
        "best_source_validation_macro_f1": selection.best_validation_macro_f1,
        "selection_history": list(selection.history),
        "selection_seconds": selection.elapsed_seconds,
        "refit_seconds": refit_seconds,
        "total_seconds": time.perf_counter() - started,
        "refit_losses": list(losses),
        "normalization_mean": normalization.mean.tolist(),
        "normalization_std": normalization.std.tolist(),
        "parameters": parameter_count(model),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "torch_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(run_directory / "metadata.json", metadata)
    return metadata
