"""Cross-device Arrival_Time alignment from rotation-invariant motion.

Activity labels are intentionally unread. The accelerometer magnitude is
binned, high-pass filtered, and restricted to its dynamic bins before
cross-correlation. A constant offset is the value added to the other device's
Arrival_Time, in milliseconds, that best matches the reference device.

Nearest-neighbour spacing after that offset describes whether both phones were
sampling densely. It does not show that the offset is correct: two continuous
clocks have a near neighbour at almost any constant shift.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from synthshift.data.alignment import nearest_indices

COARSE_BIN_MS = 50
FINE_BIN_MS = 5
FINE_RADIUS_MS = 200
DEFAULT_MAX_LAG_MS = 120_000
HIGHPASS_WINDOW_S = 2.0
DYNAMIC_PERCENTILE = 50.0
MIN_OVERLAP_S = 20.0
MIN_DYNAMIC_OVERLAP_S = 5.0
MIN_CORRELATION = 0.3
MIN_PEAK_MARGIN = 0.1
MAX_HALF_DIFF_MS = 50.0
MAX_CROSS_USER_DIFF_MS = 50.0
PEAK_SEPARATION_MS = 1_000
MAX_SPAN_MS = 6 * 60 * 60 * 1000


def prepare_stream(
    time_ms: np.ndarray, axes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | None]]:
    """Sort one accelerometer stream and reduce it to arrival time and magnitude."""

    time = pd.to_numeric(pd.Series(time_ms), errors="coerce").to_numpy(dtype=np.float64)
    values = np.asarray(axes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Accelerometer axes must have shape (n, 3)")
    finite = np.isfinite(time) & np.isfinite(values).all(axis=1)
    invalid = int((~finite).sum())
    time = time[finite]
    values = values[finite]
    reversals = int(np.sum(np.diff(time) < 0)) if time.size > 1 else 0
    order = np.argsort(time, kind="stable")
    time = time[order]
    values = values[order]
    unique = np.ones(time.shape, dtype=bool)
    if time.size > 1:
        unique[1:] = np.diff(time) > 0
    duplicates = int((~unique).sum()) if time.size else 0
    time = time[unique]
    values = values[unique]
    if time.size:
        median = float(np.median(time))
        keep = np.abs(time - median) <= MAX_SPAN_MS
        outliers = int((~keep).sum())
        time = time[keep]
        values = values[keep]
    else:
        outliers = 0
    magnitude = np.sqrt(np.sum(np.square(values), axis=1)) if time.size else np.empty(0)
    positive = np.diff(time)
    positive = positive[positive > 0]
    return (
        np.rint(time).astype(np.int64),
        magnitude.astype(np.float64),
        {
            "invalid_rows": invalid,
            "reversals": reversals,
            "duplicate_timestamps": duplicates,
            "outlier_timestamps": outliers,
            "samples": int(time.size),
            "median_dt_ms": round(float(np.median(positive)), 6) if positive.size else None,
            "span_s": round(float((time[-1] - time[0]) / 1000.0), 3) if time.size > 1 else 0.0,
        },
    )


def estimate_pair(
    reference_time_ms: np.ndarray,
    reference_magnitude: np.ndarray,
    other_time_ms: np.ndarray,
    other_magnitude: np.ndarray,
    *,
    max_lag_ms: int = DEFAULT_MAX_LAG_MS,
) -> dict[str, object]:
    """Estimate the constant Arrival_Time offset from acceleration magnitude."""

    reference_time_ms = np.asarray(reference_time_ms, dtype=np.int64)
    other_time_ms = np.asarray(other_time_ms, dtype=np.int64)
    reference_magnitude = np.asarray(reference_magnitude, dtype=np.float64)
    other_magnitude = np.asarray(other_magnitude, dtype=np.float64)
    full = _estimate_window(
        reference_time_ms,
        reference_magnitude,
        other_time_ms,
        other_magnitude,
        max_lag_ms=max_lag_ms,
    )
    halves = [None, None]
    overlap_start, overlap_end = _raw_overlap(reference_time_ms, other_time_ms)
    if overlap_end > overlap_start:
        midpoint = overlap_start + (overlap_end - overlap_start) // 2
        ranges = ((overlap_start, midpoint), (midpoint, overlap_end))
        for index, (start, end) in enumerate(ranges):
            halves[index] = _estimate_window(
                *_slice_window(
                    reference_time_ms,
                    reference_magnitude,
                    other_time_ms,
                    other_magnitude,
                    start,
                    end,
                ),
                max_lag_ms=max_lag_ms,
            )
    half_offsets = [None if half is None else half["offset_ms"] for half in halves]
    present = [offset for offset in half_offsets if offset is not None]
    half_diff = abs(present[0] - present[1]) if len(present) == 2 else None
    density_zero = _sampling_density(reference_time_ms, other_time_ms, 0)
    density_offset = _sampling_density(reference_time_ms, other_time_ms, full["offset_ms"])
    result = {
        **full,
        "half_offsets_ms": half_offsets,
        "half_offset_abs_diff_ms": None if half_diff is None else round(float(half_diff), 3),
        "sampling_density_at_zero_offset": density_zero,
        "sampling_density_after_offset": density_offset,
    }
    result["usable"] = _usable(result)
    return result


def load_accelerometer_streams(
    path: Path,
    users: set[str],
    devices: set[str],
    *,
    chunksize: int = 250_000,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, dict[str, object]]]:
    """Stream official phone accelerometer rows without reading activity labels."""

    buckets: dict[tuple[str, str], list[np.ndarray]] = {}
    usecols = ["Arrival_Time", "x", "y", "z", "User", "Device"]
    for frame in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        selected = frame.loc[frame["User"].isin(users) & frame["Device"].isin(devices)]
        for (user, device), group in selected.groupby(["User", "Device"], sort=False):
            key = (str(user), str(device))
            buckets.setdefault(key, []).append(
                group[["Arrival_Time", "x", "y", "z"]].to_numpy(dtype=np.float64)
            )
    streams = {}
    for key, parts in buckets.items():
        stacked = np.concatenate(parts)
        time_ms, magnitude, cleaning = prepare_stream(stacked[:, 0], stacked[:, 1:])
        streams[key] = (time_ms, magnitude, cleaning)
    return streams


def build_sync_report(
    streams: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, dict[str, object]]],
    config: dict[str, object],
    *,
    max_lag_ms: int = DEFAULT_MAX_LAG_MS,
) -> dict[str, object]:
    """Audit every frozen calibration pair without fitting a sensor transform."""

    devices = list(config["primary_devices"])
    fold_reports = []
    stable_pairs = []
    for fold in config["folds"]:
        users = list(fold["target_calibration_users"])
        pairs = []
        for index, reference in enumerate(devices):
            for other in devices[index + 1 :]:
                print(f"{fold['name']} {reference} -> {other}", flush=True)
                per_user = {}
                for user in users:
                    per_user[user] = _user_pair(streams, user, reference, other, max_lag_ms)
                offsets = [per_user[user]["offset_ms"] for user in users]
                if None in offsets:
                    cross_user = None
                else:
                    cross_user = round(abs(float(offsets[0] - offsets[1])), 3)
                stable = (
                    all(per_user[user]["usable"] for user in users)
                    and cross_user is not None
                    and cross_user <= MAX_CROSS_USER_DIFF_MS
                )
                pair = {
                    "reference_device": reference,
                    "other_device": other,
                    "shift_level": _shift_level(reference, other),
                    "users": per_user,
                    "cross_user_offset_abs_diff_ms": cross_user,
                    "stable_across_calibration_users": stable,
                }
                pairs.append(pair)
                stable_pairs.append(stable)
        fold_reports.append(
            {
                "name": fold["name"],
                "calibration_users": users,
                "pairs": pairs,
                "stable_pairs": int(sum(item["stable_across_calibration_users"] for item in pairs)),
            }
        )
    stream_reports = [
        {"user": user, "device": device, "cleaning": cleaning}
        for (user, device), (*_, cleaning) in sorted(streams.items())
    ]
    return {
        "method": (
            "Cross-correlation of high-passed acceleration magnitude on Arrival_Time. "
            "Labels are not read and no 3x3 map is fit."
        ),
        "parameters": {
            "coarse_bin_ms": COARSE_BIN_MS,
            "fine_bin_ms": FINE_BIN_MS,
            "fine_radius_ms": FINE_RADIUS_MS,
            "max_lag_ms": max_lag_ms,
            "highpass_window_s": HIGHPASS_WINDOW_S,
            "dynamic_percentile": DYNAMIC_PERCENTILE,
            "min_overlap_s": MIN_OVERLAP_S,
            "min_dynamic_overlap_s": MIN_DYNAMIC_OVERLAP_S,
            "min_correlation": MIN_CORRELATION,
            "min_peak_margin": MIN_PEAK_MARGIN,
            "max_half_diff_ms": MAX_HALF_DIFF_MS,
            "max_cross_user_diff_ms": MAX_CROSS_USER_DIFF_MS,
            "peak_separation_ms": PEAK_SEPARATION_MS,
        },
        "gates_fixed_before_hhar_result": True,
        "sampling_density_note": (
            "Fraction within 20 ms after applying the offset measures concurrent "
            "sampling density. A wrong constant offset still yields a near neighbour "
            "when both phones sample continuously."
        ),
        "split_config_sha256": config.get("sha256"),
        "streams": stream_reports,
        "folds": fold_reports,
        "summary": _summary(fold_reports, stable_pairs),
    }


def write_sync_report(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _user_pair(streams, user, reference, other, max_lag_ms) -> dict[str, object]:
    left = streams.get((user, reference))
    right = streams.get((user, other))
    if left is None or right is None:
        missing = [name for name, stream in ((reference, left), (other, right)) if stream is None]
        return _empty_estimate(f"missing stream: {', '.join(missing)}")
    return estimate_pair(left[0], left[1], right[0], right[1], max_lag_ms=max_lag_ms)


def _estimate_window(
    reference_time_ms: np.ndarray,
    reference_magnitude: np.ndarray,
    other_time_ms: np.ndarray,
    other_magnitude: np.ndarray,
    *,
    max_lag_ms: int,
) -> dict[str, object]:
    overlap_start, overlap_end = _raw_overlap(reference_time_ms, other_time_ms)
    overlap_s = max(0.0, (overlap_end - overlap_start) / 1000.0)
    if overlap_s < MIN_OVERLAP_S or reference_time_ms.size < 10 or other_time_ms.size < 10:
        return _empty_estimate("insufficient_overlap")
    origin = int(min(reference_time_ms[0], other_time_ms[0])) - max_lag_ms
    end = int(max(reference_time_ms[-1], other_time_ms[-1])) + max_lag_ms
    coarse = _prepare_series(
        reference_time_ms,
        reference_magnitude,
        other_time_ms,
        other_magnitude,
        origin,
        end,
        COARSE_BIN_MS,
    )
    if coarse is None:
        return _empty_estimate("empty_bins")
    reference_series, other_series = coarse
    max_lag_bins = max_lag_ms // COARSE_BIN_MS
    ranked = _ranked_lags(reference_series, other_series, max_lag_bins, COARSE_BIN_MS)
    if not ranked:
        return _empty_estimate("no_correlation_peak")
    coarse_offset = int(ranked[0][0])
    best_offset, best_correlation, best_bins = _refine_offset(
        reference_time_ms,
        reference_magnitude,
        other_time_ms,
        other_magnitude,
        origin,
        end,
        coarse_offset,
        max_lag_ms,
    )
    if best_offset is None:
        return _empty_estimate("no_correlation_peak")
    second_offset = None
    second_correlation = None
    if len(ranked) > 1:
        second_offset = int(ranked[1][0])
        second_correlation, _ = _pearson_at_lag(
            reference_series,
            other_series,
            second_offset // COARSE_BIN_MS,
            bin_ms=COARSE_BIN_MS,
        )
    margin = None
    if best_correlation is not None and second_correlation is not None:
        margin = best_correlation - second_correlation
    zero_correlation, _ = _pearson_at_lag(reference_series, other_series, 0, bin_ms=COARSE_BIN_MS)
    return {
        "status": "ok",
        "offset_ms": int(best_offset),
        "coarse_offset_ms": coarse_offset,
        "peak_correlation": _round(best_correlation),
        "correlation_at_zero": _round(zero_correlation),
        "second_peak_offset_ms": second_offset,
        "second_peak_correlation": _round(second_correlation),
        "peak_margin": _round(margin),
        "overlap_s": round(overlap_s, 3),
        "joint_bins": int(best_bins),
        "peak_at_search_boundary": abs(best_offset) >= max_lag_ms - FINE_BIN_MS,
    }


def _refine_offset(
    reference_time,
    reference_magnitude,
    other_time,
    other_magnitude,
    origin,
    end,
    coarse_offset_ms: int,
    max_lag_ms: int,
) -> tuple[int | None, float | None, int]:
    """Search 5 ms clock shifts around the 50 ms peak at lag zero."""

    start = max(-max_lag_ms, coarse_offset_ms - FINE_RADIUS_MS)
    stop = min(max_lag_ms, coarse_offset_ms + FINE_RADIUS_MS)
    start += (FINE_BIN_MS - start % FINE_BIN_MS) % FINE_BIN_MS
    best: tuple[float, int, int] | None = None
    for offset_ms in range(start, stop + 1, FINE_BIN_MS):
        shifted_time = other_time.astype(np.int64) + np.int64(offset_ms)
        prepared = _prepare_series(
            reference_time,
            reference_magnitude,
            shifted_time,
            other_magnitude,
            origin,
            end,
            COARSE_BIN_MS,
        )
        if prepared is None:
            continue
        correlation, overlap_bins = _pearson_at_lag(
            prepared[0], prepared[1], 0, bin_ms=COARSE_BIN_MS
        )
        if correlation is None:
            continue
        if best is None or correlation > best[0]:
            best = (correlation, offset_ms, overlap_bins)
    if best is None:
        return None, None, 0
    return best[1], best[0], best[2]


def _prepare_series(
    reference_time, reference_magnitude, other_time, other_magnitude, origin, end, bin_ms
):
    n_bins = int((end - origin) // bin_ms) + 1
    if n_bins <= 0 or n_bins > 5_000_000:
        return None
    reference = _dynamic(
        _highpass(_bin_means(reference_time, reference_magnitude, origin, bin_ms, n_bins), bin_ms)
    )
    other = _dynamic(
        _highpass(_bin_means(other_time, other_magnitude, origin, bin_ms, n_bins), bin_ms)
    )
    if not np.isfinite(reference).any() or not np.isfinite(other).any():
        return None
    return reference, other


def _ranked_lags(
    reference: np.ndarray, other: np.ndarray, max_lag_bins: int, bin_ms: int
) -> list[tuple[int, float]]:
    score = _fft_score(reference, other)
    zero = len(other) - 1
    start = max(0, zero - max_lag_bins)
    stop = min(len(score), zero + max_lag_bins + 1)
    lags = np.arange(start, stop) - zero
    values = score[start:stop]
    min_bins = int(MIN_DYNAMIC_OVERLAP_S * 1000 / bin_ms)
    counts = signal.correlate(
        np.isfinite(reference).astype(np.float64),
        np.isfinite(other).astype(np.float64),
        mode="full",
        method="fft",
    )
    values = np.where(counts[start:stop] >= min_bins, values, -np.inf)
    order = np.argsort(values)[::-1]
    chosen: list[tuple[int, float]] = []
    chosen_bins: list[int] = []
    radius = PEAK_SEPARATION_MS // bin_ms
    for index in order:
        if not np.isfinite(values[index]):
            continue
        lag = int(lags[index])
        if any(abs(lag - prior) <= radius for prior in chosen_bins):
            continue
        chosen_bins.append(lag)
        chosen.append((lag * bin_ms, float(values[index])))
        if len(chosen) == 3:
            break
    return chosen


def _fft_score(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    reference_filled, reference_valid = _fill_demeaned(reference)
    other_filled, other_valid = _fill_demeaned(other)
    numerator = signal.correlate(reference_filled, other_filled, mode="full", method="fft")
    reference_energy = signal.correlate(
        np.square(reference_filled), other_valid.astype(np.float64), mode="full", method="fft"
    )
    other_energy = signal.correlate(
        reference_valid.astype(np.float64), np.square(other_filled), mode="full", method="fft"
    )
    denominator = np.sqrt(np.maximum(reference_energy, 0.0) * np.maximum(other_energy, 0.0))
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, -np.inf),
        where=denominator > 1e-12,
    )


def _pearson_at_lag(
    reference: np.ndarray, other: np.ndarray, lag_bins: int, *, bin_ms: int = FINE_BIN_MS
) -> tuple[float | None, int]:
    if lag_bins >= 0:
        left = reference[lag_bins:]
        right = other[: reference.size - lag_bins]
    else:
        left = reference[: reference.size + lag_bins]
        right = other[-lag_bins:]
    if left.size == 0:
        return None, 0
    valid = np.isfinite(left) & np.isfinite(right)
    overlap = int(valid.sum())
    min_bins = int(MIN_DYNAMIC_OVERLAP_S * 1000 / bin_ms)
    if overlap < min_bins:
        return None, overlap
    x = left[valid]
    y = right[valid]
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denominator <= 1e-12:
        return None, overlap
    return float(np.dot(x, y) / denominator), overlap


def _bin_means(time_ms, values, origin, bin_ms, n_bins) -> np.ndarray:
    idx = np.floor_divide(time_ms.astype(np.int64) - np.int64(origin), np.int64(bin_ms))
    valid = (idx >= 0) & (idx < n_bins) & np.isfinite(values)
    idx = idx[valid]
    out_sum = np.zeros(n_bins, dtype=np.float64)
    out_count = np.zeros(n_bins, dtype=np.float64)
    np.add.at(out_sum, idx, values[valid])
    np.add.at(out_count, idx, 1.0)
    means = np.full(n_bins, np.nan, dtype=np.float64)
    np.divide(out_sum, out_count, out=means, where=out_count > 0)
    return means


def _highpass(values: np.ndarray, bin_ms: int) -> np.ndarray:
    window = int(round(HIGHPASS_WINDOW_S * 1000 / bin_ms))
    if window % 2 == 0:
        window += 1
    if window < 3 or window >= max(3, values.size // 2):
        finite = np.isfinite(values)
        if not finite.any():
            return values.copy()
        return np.where(finite, values - np.mean(values[finite]), np.nan)
    valid = np.isfinite(values).astype(np.float64)
    filled = np.where(np.isfinite(values), values, 0.0)
    kernel = np.ones(window)
    total = np.convolve(filled, kernel, mode="same")
    count = np.convolve(valid, kernel, mode="same")
    baseline = np.divide(total, count, out=np.full(values.size, np.nan), where=count >= 5)
    return np.where(np.isfinite(values) & np.isfinite(baseline), values - baseline, np.nan)


def _dynamic(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    threshold = float(np.percentile(np.abs(finite), DYNAMIC_PERCENTILE))
    return np.where(np.isfinite(values) & (np.abs(values) >= threshold), values, np.nan)


def _fill_demeaned(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    filled = np.zeros(values.shape, dtype=np.float64)
    if valid.any():
        filled[valid] = values[valid] - np.mean(values[valid])
    return filled, valid


def _sampling_density(reference_time, other_time, offset_ms) -> dict[str, object] | None:
    if offset_ms is None or reference_time.size == 0 or other_time.size == 0:
        return None
    shifted = other_time.astype(np.int64) + np.int64(offset_ms)
    start = max(int(reference_time[0]), int(shifted[0]))
    end = min(int(reference_time[-1]), int(shifted[-1]))
    query = reference_time[(reference_time >= start) & (reference_time <= end)]
    if query.size == 0:
        return {
            "queries": 0,
            "median_abs_dt_ms": None,
            "fraction_within_20ms": None,
            "fraction_within_50ms": None,
        }
    _, distances = nearest_indices(shifted, query)
    return {
        "queries": int(query.size),
        "median_abs_dt_ms": round(float(np.median(distances)), 3),
        "fraction_within_20ms": round(float(np.mean(distances <= 20)), 6),
        "fraction_within_50ms": round(float(np.mean(distances <= 50)), 6),
    }


def _usable(estimate: dict[str, object]) -> bool:
    margin = estimate["peak_margin"]
    half_diff = estimate["half_offset_abs_diff_ms"]
    correlation = estimate["peak_correlation"]
    return bool(
        estimate["status"] == "ok"
        and estimate["offset_ms"] is not None
        and estimate["overlap_s"] >= MIN_OVERLAP_S
        and correlation is not None
        and correlation >= MIN_CORRELATION
        and margin is not None
        and margin >= MIN_PEAK_MARGIN
        and half_diff is not None
        and half_diff <= MAX_HALF_DIFF_MS
        and not estimate["peak_at_search_boundary"]
    )


def _empty_estimate(status: str) -> dict[str, object]:
    return {
        "status": status,
        "offset_ms": None,
        "coarse_offset_ms": None,
        "peak_correlation": None,
        "correlation_at_zero": None,
        "second_peak_offset_ms": None,
        "second_peak_correlation": None,
        "peak_margin": None,
        "overlap_s": 0.0,
        "joint_bins": 0,
        "peak_at_search_boundary": False,
        "half_offsets_ms": [None, None],
        "half_offset_abs_diff_ms": None,
        "sampling_density_at_zero_offset": None,
        "sampling_density_after_offset": None,
        "usable": False,
    }


def _raw_overlap(reference_time, other_time) -> tuple[int, int]:
    if reference_time.size == 0 or other_time.size == 0:
        return 0, 0
    return (
        max(int(reference_time[0]), int(other_time[0])),
        min(int(reference_time[-1]), int(other_time[-1])),
    )


def _slice_window(reference_time, reference_magnitude, other_time, other_magnitude, start, end):
    reference_keep = (reference_time >= start) & (reference_time < end)
    other_keep = (other_time >= start) & (other_time < end)
    return (
        reference_time[reference_keep],
        reference_magnitude[reference_keep],
        other_time[other_keep],
        other_magnitude[other_keep],
    )


def _shift_level(reference: str, other: str) -> str:
    return "instance" if reference.rsplit("_", 1)[0] == other.rsplit("_", 1)[0] else "model"


def _summary(folds: list[dict[str, object]], stable_flags: list[bool]) -> dict[str, object]:
    pairs = [pair for fold in folds for pair in fold["pairs"]]
    usable = [
        user_report["usable"] for pair in pairs for user_report in pair["users"].values()
    ]
    instance = [pair for pair in pairs if pair["shift_level"] == "instance"]
    model = [pair for pair in pairs if pair["shift_level"] == "model"]

    def stable_count(group: list[dict[str, object]]) -> int:
        return int(sum(pair["stable_across_calibration_users"] for pair in group))

    return {
        "device_pairs_per_fold": len(folds[0]["pairs"]) if folds else 0,
        "fold_pair_evaluations": len(pairs),
        "usable_user_estimates": int(sum(usable)),
        "user_estimates": len(usable),
        "stable_fold_pairs": int(sum(stable_flags)),
        "stable_instance_fold_pairs": stable_count(instance),
        "instance_fold_pairs": len(instance),
        "stable_model_fold_pairs": stable_count(model),
        "model_fold_pairs": len(model),
    }


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)
