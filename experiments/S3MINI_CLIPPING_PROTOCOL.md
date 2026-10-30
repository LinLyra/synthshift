# One-sided `s3mini_1` clipping case-study protocol

**Status:** frozen exploratory case-study protocol. Commit before implementing
or scoring the clipping transform.

## Question

For transfers from a higher-range phone to `s3mini_1`, does adding a physically
fixed ±2g accelerometer clip after the committed gain transform better predict
real degradation than gain alone?

This is a severe-device case study, not a general cross-device factor test.

## Fixed evaluation cells

Evaluate exactly 12 directed cells:

- target device: `s3mini_1`;
- source devices: `nexus4_1`, `nexus4_2`, `s3_1`, and `s3_2`;
- folds: all three committed subject folds.

Transfers away from `s3mini_1` are explicitly non-evaluable for clipping,
because an already clipped source signal cannot be reconstructed. They remain
in the broader gain study but are not assigned an identity or inverse-clipping
forecast here.

## Fixed transform

Use the same trained statistical model, source test windows, calibration-user
gain profiles, feature extraction, and target evaluation as
`s1-acc-gain-v0`. Apply transforms in this order:

```text
acc_gain = acc_source * (gain_s3mini_1 / gain_source)
acc_fake = clip(acc_gain, -19.6133, +19.6133) m/s^2
gyro_fake = gyro_source
```

The rail equals `2 * 9.80665 m/s^2` and is fixed from the physical ±2g
hypothesis before scoring. No rail grid, per-axis rail, quantization, bias,
jitter, or filter transform is allowed in this experiment.

## Diagnostics

For each cell retain:

- gain-only and gain-plus-clip macro-F1 and predicted degradation;
- real target degradation;
- fraction of scalar acceleration samples changed by clipping;
- fraction of five-second windows containing at least one clipped value;
- maximum absolute value before and after clipping.

If no source-test sample exceeds the rail after gain, the cell is retained with
an exact gain-only identity and zero activation.

## Outcomes

Across the 12 cells report gain-only and clipped degradation MAE/RMSE,
Pearson/Spearman correlation, calibration slope/intercept, and mean synthetic
degradation. The primary paired outcome is:

```text
abs(real degradation - clipped prediction)
- abs(real degradation - gain-only prediction)
```

Negative values favor clipping. Report all cell-level differences and the count
improved. With only four unordered source-target device pairs, uncertainty is
descriptive; no IID significance test or strong generalization claim is allowed.

## Interpretation

- Improvement supports clipping as one contributor to the `s3mini_1` target
  domain, conditional on the current linear model.
- No improvement means the observed rail is either too rare to affect this
  model or is only a marker for other `s3mini_1` properties.
- Either result is retained. The rail must not be tuned after observing target
  scores.
