# HHAR raw sampling and timestamp audit protocol

**Status:** frozen exploratory audit. Commit before computing device profiles or
correlations with gain residuals.

## Purpose

Test whether the directional residuals left by accelerometer gain coincide with
native sampling irregularity, especially the severe `s3mini_1` fragmentation
already observed during preprocessing. This audit characterizes acquisition; it
does not yet define or evaluate a synthetic sampling perturbation.

## Data and separation

- Use original acquisition-order `Creation_Time` from the local HHAR
  accelerometer and gyroscope shards.
- Use only the two training calibration users assigned to each fold.
- Do not read activity labels, signal values, validation users, or test users.
- Compute accelerometer and gyroscope timing profiles independently.
- A fold/device profile is valid only when both calibration users have data for
  both sensors.

The current shards do not retain `Arrival_Time`; that clock is outside this
audit. Cross-phone motion alignment remains governed by the earlier sync audit.

## Per-user, per-sensor metrics

Timestamp differences are computed in original file order, never after sorting.
The following fixed metrics are retained:

1. total sample count and observed duration;
2. median positive interval and effective median sampling rate;
3. positive-interval median absolute deviation divided by the median interval;
4. positive-interval 95th and 99th percentiles;
5. fraction of all differences equal to zero;
6. fraction below zero (local reorder or reset);
7. fraction below -1 second (severe backward jump);
8. fraction above 100 ms (the committed preprocessing gap boundary);
9. forward gaps above 100 ms per observed minute;
10. fraction of positive intervals at or below 1 ms (delivery bursts).

Durations and thresholds use timestamp units inferred and asserted from the
existing nanosecond pipeline. Metrics must be computed across CSV chunks without
dropping the difference at a chunk boundary.

## Fold/device profiles

Each numeric metric is averaged with equal weight across the two calibration
users. Per-user values remain in the artifact. No user is weighted by sample
count, because the deployment unit is a participant rather than an individual
timestamp.

## Pair factors

For each of the existing 60 off-diagonal fold/source/target cells, compute the
absolute source-target difference for every continuous profile metric, separately
for accelerometer and gyroscope. Also retain directed target-minus-source
differences for median interval, normalized interval MAD, and gap rate per
minute.

## Outcomes and summaries

Reuse the committed gain residual outcomes:

- signed gain residual;
- absolute gain error.

Report descriptive Pearson and Spearman correlations for each fixed pair factor,
fold-level summaries, and profiles for all 15 fold/device combinations. No IID
p-values, multiple-testing claims, stepwise model, or outcome-fitted combination
is permitted.

## Decision rule

A timing property can motivate a synthetic perturbation only when:

1. the device-profile difference is stable across its two calibration users;
2. its residual association is directionally coherent in at least two folds;
3. it is not solely produced by transfers involving one device;
4. a causal signal transformation can be specified and frozen before scoring.

Otherwise retain the audit as a dataset limitation and proceed to quantization,
clipping, or filter-response characterization without claiming timing explains
the residual.
