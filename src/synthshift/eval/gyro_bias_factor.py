"""Frozen four-condition accelerometer-gain and gyroscope-bias experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synthshift.data.features import extract_features
from synthshift.devices.gain import apply_accelerometer_gain, estimate_gain_profile, gain_ratio
from synthshift.devices.gyro_bias import apply_gyroscope_bias, estimate_gyroscope_bias_profile
from synthshift.eval.affine_synthetic import _raw_windows, _score
from synthshift.eval.noise_synthetic import _fit
from synthshift.models.statistical_baseline import _model

BOOTSTRAP_SEED = 20261030
BOOTSTRAP_REPLICATES = 2_000


def _extended_fit(real: np.ndarray, synthetic: np.ndarray, prefix: str) -> dict[str, float]:
    result = _fit(real, synthetic, prefix)
    design = np.column_stack([synthetic, np.ones_like(synthetic)])
    slope, intercept = np.linalg.lstsq(design, real, rcond=None)[0]
    predicted = slope * synthetic + intercept
    denominator = np.sum(np.square(real - np.mean(real)))
    r_squared = 1.0 - np.sum(np.square(real - predicted)) / denominator
    result.update(
        {
            f"{prefix}_calibration_slope": float(slope),
            f"{prefix}_calibration_intercept": float(intercept),
            f"{prefix}_r_squared": float(r_squared),
        }
    )
    return result


def _subset_summary(rows: list[dict[str, object]], condition: str) -> dict[str, float]:
    real = np.asarray([row["real_degradation"] for row in rows], dtype=np.float64)
    synthetic = np.asarray(
        [row[f"{condition}_synthetic_degradation"] for row in rows], dtype=np.float64
    )
    return _extended_fit(real, synthetic, condition)


def _pair_cluster_interval(rows: list[dict[str, object]]) -> dict[str, float]:
    clusters: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = tuple(sorted((str(row["source_device"]), str(row["target_device"]))))
        combined_error = abs(
            float(row["real_degradation"])
            - float(row["combined_synthetic_degradation"])
        )
        gain_error = abs(
            float(row["real_degradation"]) - float(row["gain_synthetic_degradation"])
        )
        clusters.setdefault(key, []).append(combined_error - gain_error)
    keys = sorted(clusters)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(keys), size=len(keys))
        values = [value for item in sampled for value in clusters[keys[item]]]
        estimates[index] = np.mean(values)
    observed = float(np.mean([value for values in clusters.values() for value in values]))
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "combined_minus_gain_absolute_error": observed,
        "cluster_bootstrap_95ci_lower": float(lower),
        "cluster_bootstrap_95ci_upper": float(upper),
        "cluster_count": len(keys),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def run_gyro_bias_factor_experiment(
    features_path: Path,
    split_path: Path,
    manifest_path: Path,
    segment_root: Path,
    output_path: Path,
) -> dict[str, object]:
    with np.load(features_path) as bundle:
        table = {name: bundle[name] for name in bundle.files}
    config = json.loads(split_path.read_text(encoding="utf-8"))
    devices = config["primary_devices"]
    gain_profiles: dict[tuple[str, str], dict[str, object]] = {}
    bias_profiles: dict[tuple[str, str], dict[str, object]] = {}
    results: list[dict[str, object]] = []

    for fold in config["folds"]:
        train = np.isin(table["users"], fold["train_users"])
        validation = np.isin(table["users"], fold["validation_users"])
        test = np.isin(table["users"], fold["test_users"])
        for device in devices:
            key = (fold["name"], device)
            gain_profiles[key] = estimate_gain_profile(
                manifest_path, segment_root, fold["target_calibration_users"], device
            )
            if gain_profiles[key]["usable_users"] != fold["target_calibration_users"]:
                raise ValueError(f"Gain profile does not contain both calibration users: {key}")
            bias_profiles[key] = estimate_gyroscope_bias_profile(
                manifest_path, segment_root, fold["target_calibration_users"], device
            )

        for source in devices:
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
            indices = np.flatnonzero(test & source_mask)
            truth = table["labels"][indices]
            windows = _raw_windows(table, indices, segment_root)
            source_score = _score(model, table["features"][indices], truth)
            source_gain = float(gain_profiles[(fold["name"], source)]["gain"])
            source_bias = np.asarray(
                bias_profiles[(fold["name"], source)]["bias_rad_s"], dtype=np.float64
            )

            for target in devices:
                real_mask = test & (table["devices"] == target)
                real_score = _score(model, table["features"][real_mask], table["labels"][real_mask])
                ratio = gain_ratio(
                    source_gain, float(gain_profiles[(fold["name"], target)]["gain"])
                )
                target_bias = np.asarray(
                    bias_profiles[(fold["name"], target)]["bias_rad_s"], dtype=np.float64
                )
                bias_delta = target_bias - source_bias
                if source == target:
                    scores = {"identity": source_score, "gain": source_score,
                              "bias": source_score, "combined": source_score}
                else:
                    transformed = {
                        "gain": [apply_accelerometer_gain(window, ratio) for window in windows],
                        "bias": [apply_gyroscope_bias(window, bias_delta) for window in windows],
                        "combined": [
                            apply_gyroscope_bias(
                                apply_accelerometer_gain(window, ratio), bias_delta
                            )
                            for window in windows
                        ],
                    }
                    scores = {"identity": source_score}
                    for condition, condition_windows in transformed.items():
                        features = np.stack(
                            [extract_features(window) for window in condition_windows]
                        )
                        scores[condition] = _score(model, features, truth)
                row: dict[str, object] = {
                    "fold": fold["name"],
                    "source_device": source,
                    "target_device": target,
                    "source_macro_f1": round(source_score, 6),
                    "real_target_macro_f1": round(real_score, 6),
                    "real_degradation": round(source_score - real_score, 6),
                    "gain_ratio": ratio,
                    "gyroscope_bias_delta_rad_s": bias_delta.tolist(),
                    "calibration_users": fold["target_calibration_users"],
                }
                for condition, score in scores.items():
                    row[f"{condition}_synthetic_macro_f1"] = round(float(score), 6)
                    row[f"{condition}_synthetic_degradation"] = round(
                        source_score - float(score), 6
                    )
                results.append(row)

    shifted = [row for row in results if row["source_device"] != row["target_device"]]
    same_model = [
        row for row in shifted
        if str(row["source_device"]).rsplit("_", 1)[0]
        == str(row["target_device"]).rsplit("_", 1)[0]
    ]
    cross_model = [row for row in shifted if row not in same_model]
    summary: dict[str, object] = {
        "off_diagonal_evaluations": len(shifted),
        "same_model_evaluations": len(same_model),
        "cross_model_evaluations": len(cross_model),
        "identity_degradation_mae": float(
            np.mean(np.abs([float(row["real_degradation"]) for row in shifted]))
        ),
    }
    for condition in ("gain", "bias", "combined"):
        summary.update(_subset_summary(shifted, condition))
    summary["same_model"] = {
        condition: _subset_summary(same_model, condition)
        for condition in ("gain", "bias", "combined")
    }
    summary["cross_model"] = {
        condition: _subset_summary(cross_model, condition)
        for condition in ("gain", "bias", "combined")
    }
    summary["paired_increment"] = _pair_cluster_interval(shifted)
    summary["cells_improved_combined_vs_gain"] = sum(
        abs(float(row["real_degradation"]) - float(row["combined_synthetic_degradation"]))
        < abs(float(row["real_degradation"]) - float(row["gain_synthetic_degradation"]))
        for row in shifted
    )
    summary["same_model_cells_improved_combined_vs_gain"] = sum(
        abs(float(row["real_degradation"]) - float(row["combined_synthetic_degradation"]))
        < abs(float(row["real_degradation"]) - float(row["gain_synthetic_degradation"]))
        for row in same_model
    )
    payload = {
        "protocol_commit": "fb205ef",
        "method": "frozen four-condition accelerometer-gain and gyroscope-bias factor test",
        "summary": summary,
        "gain_profiles": list(gain_profiles.values()),
        "gyroscope_bias_profiles": list(bias_profiles.values()),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
