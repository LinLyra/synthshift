# Accelerometer-gain synthetic baseline

**Status:** exploratory positive result on one linear model. A single
rotation-invariant gain predicts real degradation better than the no-shift
null, the low-motion noise model, and the per-channel median/IQR map.
Gyroscope channels are left unchanged. No 3×3 map was fit.

## Method

For each fold, only the two training-set calibration users are used. A sample
counts as still when its gyroscope magnitude is at most 0.15 rad/s. Activity
labels are not read, and acceleration magnitude is not gated. The user gain is
the median still acceleration magnitude divided by 9.80665 m/s². The device
gain is the unweighted mean of users with at least 2,000 still samples.

The synthetic transform multiplies the three acceleration channels of each
source test window by `gain_target / gain_source`. The same logistic
feature model is then scored on the transformed windows. Test users do not
enter the gain estimate.

## Gains

The same physical phone gets nearly the same gain from different calibration
users:

| Device | Fold 0 (`b`,`g`) | Fold 1 (`a`,`h`) | Fold 2 (`c`,`i`) |
| --- | ---: | ---: | ---: |
| `nexus4_1` | 1.043 | 1.046 | 1.042 |
| `nexus4_2` | 0.999 | 1.005 | 1.000 |
| `s3_1` | 1.006 | 0.991 | 0.995 |
| `s3_2` | 0.981 | 0.968 | 0.970 |
| `s3mini_1` | 0.973 | 0.968 | 0.967 |

Cross-device gain ratios span 0.926 to 1.080. Same-device ratios are exactly 1,
and the synthetic score there matches the untouched source score.

## Prediction

On the same 60 cross-device cells, regressing real degradation on synthetic
degradation gives slope 1.040 and intercept about 0.

| Forecast | Degradation MAE | Pearson r | Spearman r | Sign agreement | Mean synthetic degradation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Predict no shift | 0.109 | — | — | — | 0.000 |
| Low-motion white noise | 0.108 | -0.003 | 0.029 | 0.65 | 0.000 |
| Median/IQR affine | 0.122 | 0.323 | 0.440 | 0.85 | 0.130 |
| Accelerometer gain | 0.067 | 0.689 | 0.660 | 0.79 | 0.054 |

Mean real degradation is 0.056. The gain forecast matches that average. The
affine forecast remains larger than the real drop and has a slightly higher
sign agreement.

The 12 same-model instance pairs are only partly explained. Their mean real
degradation is 0.042 and their mean synthetic degradation is 0.017, with MAE
0.042. The 48 cross-model pairs have mean real degradation 0.060 and mean
synthetic degradation 0.063, with MAE 0.074. Scalar acceleration gain tracks
model-level shift and leaves a residual instance-level shift.

Machine-readable result: `results/reviewed/gain_synthetic_baseline.json`.
