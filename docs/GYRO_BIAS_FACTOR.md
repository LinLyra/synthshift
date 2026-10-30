# Gyroscope-bias factor experiment

**Status:** exploratory mixed result on one linear model. The primary protocol
was frozen in commit `fb205ef`, and the implementation was committed as
`2a54414` before this result was generated.

## Design

The experiment evaluates the same 60 off-diagonal HHAR cells under four fixed
conditions: identity, accelerometer gain only, gyroscope bias only, and gain
plus bias. Each device profile uses both training-set calibration users. Each
user contributes at least 2,000 samples passing the precommitted acceleration-
magnitude and gyroscope-norm gate. Labels and test users do not enter profile
estimation.

The per-axis gyroscope transform is `gyro + (bias_target - bias_source)`. The
gain transform is unchanged from `s1-acc-gain-v0`.

## Primary result

| Forecast | MAE | Pearson r | Spearman r | Mean synthetic degradation |
| --- | ---: | ---: | ---: | ---: |
| Identity | 0.1087 | — | — | 0.0000 |
| Gain only | 0.0675 | 0.689 | 0.660 | 0.0540 |
| Bias only | 0.1035 | 0.209 | 0.326 | 0.0028 |
| Gain + bias | 0.0648 | 0.708 | 0.695 | 0.0600 |

The combined condition reduces mean absolute degradation error by 0.00266
relative to gain only. A precommitted bootstrap over the 10 unordered physical
device-pair clusters gives a 95% percentile interval of [-0.00744, 0.00098].
The interval includes zero, so the overall incremental improvement is not yet
reliable evidence of a general additional factor. Combined error is smaller in
38 of 60 cells, equal in two, and larger in 20.

The combined calibration regression has slope 1.061, intercept -0.0077, and
R-squared 0.501, compared with slope 1.040, intercept approximately zero, and
R-squared 0.475 for gain only.

## Same-model residual

| Subset and forecast | Cells | MAE | Pearson r | Mean synthetic degradation |
| --- | ---: | ---: | ---: | ---: |
| Same-model, gain | 12 | 0.0421 | 0.588 | 0.0172 |
| Same-model, gain + bias | 12 | 0.0332 | 0.742 | 0.0242 |
| Cross-model, gain | 48 | 0.0738 | 0.701 | 0.0631 |
| Cross-model, gain + bias | 48 | 0.0727 | 0.712 | 0.0690 |

The same-model average improves, but only five of 12 same-model cells improve,
two are unchanged, and five worsen. The mean improvement is concentrated in
Galaxy S3 instance transfers. Nexus 4 instance transfers are unchanged or
slightly worse. The largest improvements are `s3_1 -> s3_2` in folds 1 and 0
(-0.0483 and -0.0262 absolute-error change). This is not evidence that bias
uniformly explains instance-level shift.

## Profile stability warning

The two calibration users sometimes disagree materially. The largest absolute
per-axis disagreement is 0.0347 rad/s for fold 0 `s3_2`; the corresponding
device-profile components are near 0.010, -0.010, and 0.010 rad/s. Nexus 4
profiles are much closer to zero and more stable. A low-motion activity segment
can contain posture- or behavior-dependent angular motion, so the median is not
automatically a pure hardware offset.

The planned threshold sensitivity analysis must therefore be treated as a
stability audit, not as permission to select the threshold with the best target
score. A stronger next estimator should add temporal-stability criteria using
calibration data only.

## Preplanned gate sensitivity

The protocol-fixed 0.05, 0.10, and 0.20 rad/s checks were run only after the
0.15 primary result was committed. They do not replace the primary threshold.

| Gyroscope gate | Combined MAE | Combined Pearson r | Same-model MAE | Increment 95% CI |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 rad/s | 0.06479 | 0.70815 | 0.03330 | [-0.00753, 0.00096] |
| 0.10 rad/s | 0.06478 | 0.70814 | 0.03323 | [-0.00750, 0.00093] |
| **0.15 rad/s primary** | **0.06483** | **0.70801** | **0.03323** | **[-0.00744, 0.00098]** |
| 0.20 rad/s | 0.06482 | 0.70812 | 0.03318 | [-0.00745, 0.00097] |

The substantive conclusion is invariant across this range: combined gain and
bias is slightly better on average than gain alone, but the clustered interval
includes zero. The near-identical results also show that the mixed finding is
not caused by selecting 0.15 rad/s after inspecting target performance.

Machine-readable result: `results/reviewed/gyro_bias_factor.json`.
Sensitivity artifacts use the suffixes `threshold_005`, `threshold_010`, and
`threshold_020` in the same directory.
