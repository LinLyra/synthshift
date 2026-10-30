# Low-motion spectral-shape audit protocol

**Status:** frozen exploratory audit. Commit before computing spectral profiles
or correlations with gain residuals.

## Purpose

Determine whether calibration-only spectral shape and temporal correlation can
motivate a colored-noise or filter-response experiment after white-noise scale,
scalar gain, gyroscope bias, and generic timing factors proved incomplete.

This audit does not identify a hardware transfer function: the phones were not
reliably sample-aligned, and their true mechanical input is unknown. It only
tests whether reproducible low-motion spectral signatures coincide with gain
residuals.

## Data separation

- Use the committed anti-aliased 50 Hz segment corpus.
- Use only both training calibration users assigned to each fold.
- Do not read activity labels, validation users, test users, or target scores
  while constructing device profiles.
- Analyze accelerometer and gyroscope separately.
- A profile is invalid unless both calibration users meet the window minimum.

## Low-motion windows

A sample is eligible under the already committed gate:

```text
abs(norm(acceleration) - 9.80665) <= 0.5 m/s^2
norm(gyroscope) <= 0.15 rad/s
```

Within each legal preprocessing segment, identify contiguous eligible runs.
Split each run into non-overlapping 128-sample windows (2.56 seconds). Do not
join runs, segments, or activity boundaries. Require at least 10 windows per
calibration user/device.

## Fixed per-window features

For each of the three channels of one sensor:

1. subtract the window mean;
2. apply a Hann window;
3. compute the one-sided periodogram with the correct window-energy
   normalization;
4. exclude DC and frequencies above 18 Hz.

Retain these shape features:

- fraction of 0.5–18 Hz power in 0.5–2, 2–5, 5–10, and 10–18 Hz bands;
- spectral centroid over 0.5–18 Hz;
- log10 ratio of 10–18 Hz power to 2–5 Hz power, with a fixed numerical epsilon;
- lag-one autocorrelation of the demeaned channel.

Average the three channel values within a window to reduce axis-orientation
dependence. For each user/device/sensor, take the median across windows. The
fold/device profile is the unweighted mean of its two user profiles. Per-user
values and window counts remain in the artifact.

## Pair factors and outcomes

For each of the 60 existing off-diagonal cells, compute both the absolute
source-target difference and target-minus-source difference for every spectral
feature, separately for accelerometer and gyroscope.

Reuse the committed outcomes:

- signed gain residual;
- absolute gain error.

Report descriptive Pearson and Spearman correlations overall and within each
fold. Do not report IID p-values or fit an outcome-optimized feature
combination.

## Stability and decision rule

A feature may motivate a filter or colored-noise transform only if:

1. the two calibration users give the same device ordering or closely similar
   values;
2. its residual association has a coherent direction in at least two folds;
3. it is not solely an indicator for `s3mini_1`;
4. the proposed signal transform can be specified without target labels and
   frozen before evaluation.

The 20 Hz preprocessing low-pass is shared by all devices and limits claims
about native response near Nyquist. Any positive result is a property of the
retained 0.5–18 Hz corpus, not a recovered manufacturer frequency response.
