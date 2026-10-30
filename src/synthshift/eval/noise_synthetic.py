"""Exploratory noise-only synthetic-to-real experiment."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr

from synthshift.data.features import extract_features
from synthshift.devices.noise import (
    ACCELERATION_RADIUS_M_S2,
    GRAVITY_M_S2,
    GYRO_MAX_RAD_S,
    MIN_DIFFERENCE_PAIRS,
    added_noise_scale,
    apply_white_noise,
    estimate_noise_profile,
)
from synthshift.eval.affine_synthetic import _raw_windows, _score
from synthshift.models.statistical_baseline import _model

REPEATS = 5
SEED = 0


def run_noise_experiment(
    features_path: Path,
    split_path: Path,
    manifest_path: Path,
    segment_root: Path,
    output_path: Path,
    *,
    affine_path: Path | None = None,
    still_report: dict[str, object] | None = None,
    repeats: int = REPEATS,
) -> dict[str, object]:
    with np.load(features_path) as bundle:
        table = {name: bundle[name] for name in bundle.files}
    config = json.loads(split_path.read_text(encoding="utf-8"))
    devices = config["primary_devices"]
    profile_cache = {}
    results = []

    for fold_index, fold in enumerate(config["folds"]):
        train = np.isin(table["users"], fold["train_users"])
        validation = np.isin(table["users"], fold["validation_users"])
        test = np.isin(table["users"], fold["test_users"])
        for device in devices:
            profile_cache[(fold["name"], device)] = estimate_noise_profile(
                manifest_path,
                segment_root,
                fold["target_calibration_users"],
                device,
            )

        for source_index, source in enumerate(devices):
            print(f"{fold['name']} {source}", flush=True)
            source_mask = table["devices"] == source
            best_c, best_validation = None, -1.0
            for c_value in (0.1, 1.0, 10.0):
                candidate = _model(c_value)
                candidate.fit(
                    table["features"][train & source_mask], table["labels"][train & source_mask]
                )
                score = _score(
                    candidate,
                    table["features"][validation & source_mask],
                    table["labels"][validation & source_mask],
                )
                if score > best_validation:
                    best_c, best_validation = c_value, score

            model = _model(float(best_c))
            fit = (train | validation) & source_mask
            model.fit(table["features"][fit], table["labels"][fit])
            source_test_indices = np.flatnonzero(test & source_mask)
            source_truth = table["labels"][source_test_indices]
            source_score = _score(model, table["features"][source_test_indices], source_truth)
            source_windows = _raw_windows(table, source_test_indices, segment_root)
            source_sigma = np.asarray(profile_cache[(fold["name"], source)]["sigma"])

            for target_index, target in enumerate(devices):
                target_test = test & (table["devices"] == target)
                real_score = _score(
                    model, table["features"][target_test], table["labels"][target_test]
                )
                scale = added_noise_scale(
                    source_sigma, np.asarray(profile_cache[(fold["name"], target)]["sigma"])
                )
                if np.all(scale == 0):
                    scores = [source_score] * repeats
                else:
                    scores = [
                        _repeat_score(
                            model,
                            source_windows,
                            source_truth,
                            scale,
                            seed=(SEED, fold_index, source_index, target_index, repeat),
                        )
                        for repeat in range(repeats)
                    ]
                synthetic_score = float(np.mean(scores))
                results.append(
                    {
                        "fold": fold["name"],
                        "source_device": source,
                        "target_device": target,
                        "source_macro_f1": round(source_score, 6),
                        "real_target_macro_f1": round(real_score, 6),
                        "synthetic_target_macro_f1": round(synthetic_score, 6),
                        "synthetic_score_std": round(float(np.std(scores)), 6),
                        "real_degradation": round(source_score - real_score, 6),
                        "synthetic_degradation": round(source_score - synthetic_score, 6),
                        "degradation_error": round(
                            (source_score - synthetic_score) - (source_score - real_score), 6
                        ),
                        "added_sigma": scale.tolist(),
                        "calibration_users": fold["target_calibration_users"],
                    }
                )

    payload = {
        "method": (
            "unlabelled low-motion difference noise; one-sided white Gaussian; "
            "still-experiment files are not used for instance scales"
        ),
        "parameters": {
            "repeats": repeats,
            "seed": SEED,
            "gravity_m_s2": GRAVITY_M_S2,
            "acceleration_radius_m_s2": ACCELERATION_RADIUS_M_S2,
            "gyro_max_rad_s": GYRO_MAX_RAD_S,
            "min_difference_pairs": MIN_DIFFERENCE_PAIRS,
        },
        "still_identity": still_report,
        "summary": _summary(results, affine_path),
        "profiles": list(profile_cache.values()),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _repeat_score(model, windows, labels, scale, seed: tuple[int, ...]) -> float:
    rng = np.random.default_rng(seed)
    transformed = np.stack([apply_white_noise(window, scale, rng) for window in windows])
    features = np.stack([extract_features(window) for window in transformed])
    return _score(model, features, labels)


def _summary(results: list[dict[str, object]], affine_path: Path | None) -> dict[str, object]:
    shifted = [row for row in results if row["source_device"] != row["target_device"]]
    real = np.asarray([row["real_degradation"] for row in shifted], dtype=np.float64)
    synthetic = np.asarray([row["synthetic_degradation"] for row in shifted], dtype=np.float64)
    summary = {
        "off_diagonal_evaluations": len(shifted),
        **_fit(real, synthetic, "noise"),
        "null_degradation_mae": float(np.mean(np.abs(real))),
    }
    if affine_path is not None:
        affine = json.loads(affine_path.read_text(encoding="utf-8"))
        affine_rows = {
            (row["fold"], row["source_device"], row["target_device"]): row
            for row in affine["results"]
            if row["source_device"] != row["target_device"]
        }
        affine_degradation = np.asarray(
            [
                affine_rows[(row["fold"], row["source_device"], row["target_device"])][
                    "synthetic_degradation"
                ]
                for row in shifted
            ],
            dtype=np.float64,
        )
        summary.update(_fit(real, affine_degradation, "affine"))
    return summary


def _fit(real: np.ndarray, synthetic: np.ndarray, prefix: str) -> dict[str, float | None]:
    error = synthetic - real
    nonzero = (synthetic != 0) & (real != 0)
    sign_agreement = (
        float(np.mean(np.sign(synthetic[nonzero]) == np.sign(real[nonzero])))
        if np.any(nonzero)
        else None
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        pearson = pearsonr(real, synthetic).statistic
        spearman = spearmanr(real, synthetic).statistic
    return {
        f"{prefix}_degradation_mae": _finite(np.mean(np.abs(error))),
        f"{prefix}_degradation_rmse": _finite(np.sqrt(np.mean(np.square(error)))),
        f"{prefix}_pearson_r": _finite(pearson),
        f"{prefix}_spearman_r": _finite(spearman),
        f"{prefix}_sign_agreement": _finite(sign_agreement),
        f"{prefix}_mean_synthetic_degradation": _finite(np.mean(synthetic)),
    }


def _finite(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None
