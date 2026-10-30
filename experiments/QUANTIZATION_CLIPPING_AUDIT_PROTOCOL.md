# Raw quantization and clipping-signature audit protocol

**Status:** frozen exploratory audit. Commit before computing full calibration
profiles or residual correlations.

## Purpose and claim boundary

Measure reproducible discreteness, repeated-value, and extreme-value signatures
in raw HHAR values. The CSV export and Android sensor stack may transform native
ADC codes, so the audit will not call an empirical spacing a hardware LSB or an
observed extreme a manufacturer range without external metadata.

## Data separation

- Use raw accelerometer and gyroscope shard values in acquisition order.
- Use only both training calibration users assigned to each fold.
- Do not read activity labels, timestamps, validation users, or test users.
- Profile each channel separately, then use fixed rotation-robust channel
  summaries.

## Fixed per-channel metrics

For each user/device/sensor/channel:

1. sample count and unique-value fraction;
2. consecutive exact-repeat fraction;
3. 10th and 50th percentiles of positive spacings between sorted unique values;
4. observed minimum and maximum;
5. exact-minimum and exact-maximum occupancy;
6. occupancy within one empirical 10th-percentile spacing of either extreme;
7. longest consecutive run at either exact extreme;
8. 0.01st and 99.99th percentiles and their robust range;
9. extreme symmetry error: absolute difference between positive and negative
   absolute rails divided by their mean magnitude.

The positive-spacing calculation uses exact exported floating values. It is an
empirical representation signature, not a claim about analog resolution.

## Fixed sensor summaries

Within a user/device/sensor profile:

- take the median across three channels for unique fraction, repeat fraction,
  spacing percentiles, robust range, and symmetry error;
- take the maximum across channels for exact-extreme occupancy,
  within-one-spacing occupancy, and longest extreme run.

The fold/device profile is the unweighted mean of its two user profiles.
Per-channel and per-user values remain in the artifact.

## Pair factors and outcomes

For every summary metric, construct absolute source-target difference and
target-minus-source difference for the existing 60 off-diagonal cells,
separately for accelerometer and gyroscope. Reuse signed gain residual and
absolute gain error as descriptive outcomes.

Report overall and fold-specific Pearson/Spearman correlations without IID
p-values or outcome-fitted combinations.

## Decision rule

A quantization or clipping transform may be proposed only when:

1. both calibration users show a stable device signature;
2. the signature is repeated across folds;
3. its residual direction is coherent in at least two folds;
4. it is not merely an activity-amplitude difference;
5. transform parameters and saturation semantics are frozen before scoring.

If only `s3mini_1` shows an apparent ±2g rail, retain it as a severe-device
clipping case study rather than a generic cross-device calibration factor.
