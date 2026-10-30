# Gain-residual device-factor audit protocol

**Status:** frozen exploratory audit. This document must be committed before
the audit table or correlations are generated.

## Purpose

Identify which already-measured, calibration-only device properties are
associated with the prediction error left by the scalar accelerometer-gain
baseline. This audit selects the next physically motivated experiment; it does
not establish causality and does not optimize a new synthetic transform.

## Inputs

Only committed artifacts are used:

- gain profiles and 60 off-diagonal results from `s1-acc-gain-v0`;
- low-motion noise profiles from `s1-noise-low-motion-v0`;
- primary 0.15 rad/s gyroscope-bias profiles from
  `s1-gyro-bias-factor-v0`.

All profiles were estimated from the two training calibration users. No new
labels, validation users, test-user signals, or target scores may be used to
construct a device factor. Real target degradation is used only as the audit
outcome.

## Fixed outcomes

For every fold/source/target cell:

```text
signed_gain_residual = real_degradation - gain_synthetic_degradation
absolute_gain_error = abs(signed_gain_residual)
```

Positive signed residual means gain under-predicts degradation; negative means
it over-predicts degradation.

## Fixed calibration-side factors

The following factors are computed before inspecting their correlations:

1. `absolute_log_gain_ratio`: absolute log target/source acceleration gain;
2. `gyroscope_bias_delta_l2`: L2 norm of target minus source bias;
3. `gyroscope_bias_user_disagreement_max`: larger of the source and target L2
   differences between their two calibration-user bias estimates;
4. `noise_sigma_delta_l2`: L2 norm of target minus source six-channel noise
   scale;
5. `one_sided_added_noise_l2`: L2 norm of the committed one-sided noise scale
   needed to raise source variance toward target variance;
6. `same_model`: one for Nexus 4 instance pairs or Galaxy S3 instance pairs,
   otherwise zero.

No factor is removed because it has low variance or an unfavorable result.

## Fixed summaries

- Pearson and Spearman correlation of every continuous factor with both fixed
  outcomes across the 60 cells;
- mean and median outcomes for `same_model` and `cross_model` cells;
- the five largest positive and five largest negative signed residual cells;
- fold-level and unordered-device-pair-level descriptive means.

Correlations are descriptive because cells share folds, trained models, and
physical devices. No ordinary IID p-values will be reported. No multivariable
model, stepwise selection, or threshold search will be performed in this audit.

## Decision rule for the next experiment

A factor can motivate a next perturbation only if all conditions hold:

1. it has a physically meaningful transform that can be applied without test
   labels;
2. its association is not confined to a single fold or one directed cell;
3. calibration-user stability is adequate for the intended claim;
4. the proposed transform and parameter range are frozen in a new protocol
   before target performance is evaluated.

If no factor satisfies these conditions, the next step is a raw timing and
sampling audit rather than combining weak factors into an outcome-fitted model.
