"""Frozen checkpoint integrity and target evaluation for neural models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from synthshift.models.neural import build_neural_model
from synthshift.train.jobs import PROTOCOL_COMMIT, NeuralJob, completed_metadata, resolve_device
from synthshift.train.neural import ChannelNormalization


def audit_formal_checkpoints(
    jobs: list[NeuralJob], output_root: Path
) -> dict[str, object]:
    complete, missing, invalid = [], [], []
    for job in jobs:
        directory = output_root / "formal" / job.job_id
        try:
            metadata = completed_metadata(directory)
        except (ValueError, json.JSONDecodeError, OSError) as error:
            invalid.append({"job_id": job.job_id, "error": str(error)})
            continue
        if metadata is None:
            missing.append(job.job_id)
            continue
        if (
            metadata.get("smoke")
            or metadata.get("job") != job.__dict__
            or metadata.get("protocol_commit") != PROTOCOL_COMMIT
        ):
            invalid.append({"job_id": job.job_id, "error": "metadata identity mismatch"})
            continue
        complete.append(job.job_id)
    return {
        "expected": len(jobs),
        "complete": len(complete),
        "missing": missing,
        "invalid": invalid,
        "ready_for_target_evaluation": len(complete) == len(jobs) and not invalid,
    }


def _predict_indices(
    model: torch.nn.Module,
    windows: np.ndarray,
    normalization: ChannelNormalization,
    device: torch.device,
    *,
    acceleration_ratio: float = 1.0,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            values = np.asarray(windows[start : start + batch_size], dtype=np.float32).copy()
            values[:, :, :3] *= acceleration_ratio
            values = (values - normalization.mean) / normalization.std
            tensor = torch.from_numpy(np.ascontiguousarray(values.transpose(0, 2, 1))).to(device)
            predictions.append(model(tensor).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64)


def score_predictions(
    truth: np.ndarray, predictions: np.ndarray, users: np.ndarray
) -> dict[str, float]:
    if not (len(truth) == len(predictions) == len(users)):
        raise ValueError("Truth, predictions, and users must be row-aligned")
    if len(truth) == 0:
        raise ValueError("Cannot score an empty evaluation set")
    pooled = f1_score(truth, predictions, average="macro", zero_division=0)
    per_subject = [
        f1_score(
            truth[users == user],
            predictions[users == user],
            average="macro",
            zero_division=0,
        )
        for user in sorted(np.unique(users))
    ]
    return {"pooled_macro_f1": float(pooled), "subject_macro_f1": float(np.mean(per_subject))}


def _profile_gain(
    profiles: list[dict[str, object]], calibration_users: list[str], device: str
) -> float:
    matches = [
        profile
        for profile in profiles
        if profile["device"] == device and profile["users"] == calibration_users
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one gain profile for device={device}, users={calibration_users}"
        )
    if matches[0]["usable_users"] != calibration_users:
        raise ValueError("Gain profile does not contain both frozen calibration users")
    return float(matches[0]["gain"])


def evaluate_formal_job(
    job: NeuralJob,
    *,
    features_path: Path,
    raw_cache_path: Path,
    split_path: Path,
    gain_artifact_path: Path,
    output_root: Path,
    device_name: str = "cpu",
) -> list[dict[str, object]]:
    run_directory = output_root / "formal" / job.job_id
    metadata = completed_metadata(run_directory)
    if metadata is None or metadata.get("smoke"):
        raise ValueError(f"Formal checkpoint is incomplete: {job.job_id}")
    if metadata.get("job") != job.__dict__:
        raise ValueError("Checkpoint metadata does not match requested job")
    if metadata.get("protocol_commit") != PROTOCOL_COMMIT:
        raise ValueError("Checkpoint metadata protocol commit mismatch")
    device = resolve_device(device_name)
    checkpoint = torch.load(run_directory / "checkpoint.pt", map_location="cpu")
    if checkpoint["job"] != job.__dict__:
        raise ValueError("Checkpoint payload does not match requested job")
    if checkpoint.get("protocol_commit") != PROTOCOL_COMMIT:
        raise ValueError("Checkpoint payload protocol commit mismatch")
    classes = np.asarray(checkpoint["classes"])
    normalization = ChannelNormalization(
        mean=np.asarray(checkpoint["normalization_mean"], dtype=np.float32),
        std=np.asarray(checkpoint["normalization_std"], dtype=np.float32),
    )
    model = build_neural_model(job.architecture, len(classes))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)

    with np.load(features_path) as bundle:
        users = bundle["users"]
        devices = bundle["devices"]
        labels = bundle["labels"]
    raw = np.load(raw_cache_path, mmap_mode="r")
    if len(raw) != len(labels):
        raise ValueError("Raw cache and feature metadata are not aligned")
    config = json.loads(split_path.read_text(encoding="utf-8"))
    fold = next(item for item in config["folds"] if item["name"] == job.fold)
    gain_artifact = json.loads(gain_artifact_path.read_text(encoding="utf-8"))
    profiles = gain_artifact["profiles"]
    test = np.isin(users, fold["test_users"])
    source_indices = np.flatnonzero(test & (devices == job.source_device))
    source_prediction_indices = _predict_indices(
        model, raw[source_indices], normalization, device
    )
    source_predictions = classes[source_prediction_indices]
    source_scores = score_predictions(
        labels[source_indices], source_predictions, users[source_indices]
    )
    source_gain = _profile_gain(
        profiles, fold["target_calibration_users"], job.source_device
    )

    rows = []
    for target in config["primary_devices"]:
        target_indices = np.flatnonzero(test & (devices == target))
        real_indices = _predict_indices(model, raw[target_indices], normalization, device)
        real_scores = score_predictions(
            labels[target_indices], classes[real_indices], users[target_indices]
        )
        target_gain = _profile_gain(profiles, fold["target_calibration_users"], target)
        ratio = target_gain / source_gain
        synthetic_indices = _predict_indices(
            model,
            raw[source_indices],
            normalization,
            device,
            acceleration_ratio=ratio,
        )
        synthetic_scores = score_predictions(
            labels[source_indices], classes[synthetic_indices], users[source_indices]
        )
        row = {
            "fold": job.fold,
            "source_device": job.source_device,
            "target_device": target,
            "architecture": job.architecture,
            "seed": job.seed,
            "gain_ratio": ratio,
            "checkpoint_sha256": metadata["checkpoint_sha256"],
        }
        for score_name, source_score in source_scores.items():
            prefix = "pooled" if score_name == "pooled_macro_f1" else "subject"
            real_score = real_scores[score_name]
            synthetic_score = synthetic_scores[score_name]
            row.update(
                {
                    f"source_{prefix}_macro_f1": source_score,
                    f"real_target_{prefix}_macro_f1": real_score,
                    f"synthetic_target_{prefix}_macro_f1": synthetic_score,
                    f"real_{prefix}_degradation": source_score - real_score,
                    f"synthetic_{prefix}_degradation": source_score - synthetic_score,
                }
            )
        rows.append(row)
    return rows
