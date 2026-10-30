# Gain-residual calibration-factor audit

**Status:** exploratory descriptive audit. Protocol commit `cc76dd4`; audit
implementation commit `e93ac4a`. No multivariable model or factor selection was
fit to the target outcomes.

## Residual structure

The outcome is real degradation minus gain-predicted degradation. Positive
values mean gain under-predicts the real drop; negative values mean it
over-predicts the drop.

| Subset | Cells | Mean signed residual | Median signed residual | Mean absolute error |
| --- | ---: | ---: | ---: | ---: |
| Same phone model | 12 | 0.0245 | 0.0030 | 0.0421 |
| Different models | 48 | -0.0035 | 0.0122 | 0.0738 |

Fold 0 has mean absolute error 0.1067, compared with 0.0440 and 0.0517 in
folds 1 and 2. Its four largest negative residuals all use `s3mini_1` as the
source; three of the largest positive residuals use `s3mini_1` as the target.
This is a strong directional domain-transfer asymmetry, not a uniform device
offset.

## Fixed factor correlations

| Calibration-only factor | Signed residual Pearson | Signed Spearman | Absolute error Pearson | Absolute Spearman |
| --- | ---: | ---: | ---: | ---: |
| Absolute log gain ratio | -0.055 | -0.091 | -0.058 | -0.121 |
| Gyroscope bias delta L2 | -0.056 | -0.051 | 0.226 | 0.275 |
| Maximum bias user disagreement | 0.104 | 0.073 | 0.217 | 0.118 |
| Noise-scale delta L2 | -0.215 | -0.189 | 0.602 | 0.293 |
| One-sided added-noise L2 | -0.667 | -0.524 | 0.385 | 0.279 |

The one-sided noise factor encodes direction: it is zero when the target is
quieter than the source and positive in the reverse direction. Its signed
correlation is therefore not evidence that white noise causes the residual.
The committed perturbation experiment already showed that adding the estimated
white noise changes macro-F1 by at most 0.007 and does not predict real drops.

The noise-delta Pearson correlation with absolute error is also not robust.
In a post-hoc localization check, removing every transfer involving
`s3mini_1` changes it from 0.602 to -0.127; its fold-specific Pearson values are
0.820, 0.071, and 0.113. This check is diagnostic rather than a new
pre-registered result.

## Decision

No current factor satisfies the protocol rule for motivating another calibrated
perturbation:

- acceleration gain has already been applied and its magnitude is unrelated to
  the remaining error;
- gyroscope bias associations are weak and the factor experiment's clustered
  interval includes zero;
- noise associations are directional or concentrated in fold 0 transfers
  involving `s3mini_1`, while direct noise injection is a null result.

The next step is therefore a raw sampling and timing audit. It should determine
whether `s3mini_1` and the high-error directions differ in native interval
distribution, gaps, delivery bursts, reorder rate, or effective filtering
before any sampling-jitter transform is proposed.

Machine-readable artifact:
`results/reviewed/gain_residual_factor_audit.json`.
