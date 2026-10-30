# Gyroscope-bias factor protocol

**Status:** frozen exploratory protocol. This document contains no gyroscope-
bias results and must be committed before the implementation is run.

## Question

Does an unlabelled, calibration-user estimate of per-axis gyroscope bias explain
cross-device degradation left unexplained by the scalar accelerometer-gain
transform, particularly between two instances of the same phone model?

This is a factor test, not a search for the perturbation that minimizes error.
All four conditions below will be retained, including null or adverse results.

## Data separation

- Dataset, five primary phone instances, subject folds, preprocessing, feature
  model, and hyperparameter selection remain identical to
  `s1-acc-gain-v0`.
- Only both named `target_calibration_users` in a fold may estimate a device
  profile. A profile is unavailable unless both users pass the sample-count
  rule.
- Activity labels are not read by the profile estimator.
- Validation and test users never enter a device profile.
- The real target test score is used only after every profile and transform has
  been fixed.

## Primary low-motion estimator

A sample is eligible when both conditions hold:

1. `abs(norm(acceleration) - 9.80665) <= 0.5 m/s^2`;
2. `norm(gyroscope) <= 0.15 rad/s`.

For each calibration user and device, the three-axis bias is the componentwise
median gyroscope vector over eligible samples. At least 2,000 eligible samples
are required per user. The device bias is the unweighted mean of the two user
bias vectors. Per-user estimates, eligible counts, and their componentwise
difference must be stored in the result artifact.

The norm condition can weakly truncate very large biases. It is retained here
because 0.15 rad/s is substantially above the observed low-motion channel
scales and because changing the gate after seeing results would add researcher
degrees of freedom.

## Fixed transforms and four conditions

Let `b_s` and `b_t` be the source and target device bias vectors. The bias-only
synthetic transform is:

```text
gyro_fake = gyro_source + (b_t - b_s)
acc_fake = acc_source
```

The scalar acceleration gain is exactly the committed `s1-acc-gain-v0`
transform. Every off-diagonal fold/source/target cell will be evaluated under:

1. identity control;
2. accelerometer gain only;
3. gyroscope bias only;
4. accelerometer gain plus gyroscope bias.

For the combined condition, acceleration and gyroscope transforms act on
disjoint channels, so their order is immaterial. Same-device transforms must be
exact identities and serve as an implementation invariant.

## Fixed evaluation sets

- Primary: all 60 directed off-diagonal fold/source/target cells.
- Same-model residual subset: the 12 directed cells formed by
  `nexus4_1`/`nexus4_2` and `s3_1`/`s3_2` over three folds.
- Cross-model subset: the remaining 48 cells.

No cell may be excluded because its estimated bias or score is unfavorable.
If a required profile is unavailable, the experiment is incomplete rather than
silently reduced to fewer cells.

## Outcomes

For bias-only and gain-plus-bias predictions, report degradation MAE, RMSE,
Pearson correlation, Spearman correlation, sign agreement, mean synthetic
degradation, calibration slope/intercept, and R-squared. Report the same-model
and cross-model subset summaries separately.

The primary incremental contrast is paired cell-level error:

```text
abs(real degradation - combined synthetic degradation)
- abs(real degradation - gain-only synthetic degradation)
```

Negative values favor the combined transform. Also report how many of the 60
cells and how many of the 12 same-model cells improve; do not use that count as
a significance test.

For uncertainty, resample the 10 unordered physical device pairs as clusters,
retaining both directions and all three folds within every sampled cluster.
Use 2,000 bootstrap replicates with seed 20261030 and percentile 95% intervals.
The small number of clusters must be stated as a limitation.

## Sensitivity analyses

After the primary result is frozen, repeat profile estimation at gyroscope-norm
thresholds 0.05, 0.10, and 0.20 rad/s. These are robustness checks only: the
0.15 rad/s result remains primary, and no threshold will be selected by target
test performance.

## Interpretation policy

- Improvement of the combined condition supports complementary acceleration-
  gain and gyroscope-bias effects for this model.
- Bias-only agreement without incremental combined improvement indicates
  overlapping predictive structure rather than an additional explained factor.
- A null or adverse result is retained and narrows the plausible explanation
  of the same-model residual.
- No result from this experiment establishes model-ranking validity; that
  requires the preregistered heterogeneous multi-model study.
