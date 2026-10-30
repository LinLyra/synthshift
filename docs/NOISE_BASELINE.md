# Low-motion noise synthetic baseline

**Status:** exploratory null result. White noise matched to still-sample scales
does not predict real cross-device degradation. No 3×3 map was fit.

## Still-experiment identity

The still experiment records accelerometers resting in six orientations. Its
folders are lab device names (`it-151`, `3Renault-AH`), not the activity
instance IDs. The only shared string is `nexus4_2`. That folder sits beside
eleven other Nexus 4 names, three Galaxy S3 names, and one Galaxy S3 Mini name.
A shared string does not identify the activity phones, so still files were not
used to set instance noise.

## Method

Noise scales come from the anti-aliased 50 Hz corpus. For each fold, only the
two training-set calibration users are read. A sample pair contributes only
when both samples have acceleration magnitude within 0.5 m/s² of 9.80665 and
gyroscope magnitude at most 0.15 rad/s. Activity labels are not read.
Differences are taken inside each segment. Each user needs at least 2,000 such
pairs. The reported scale is the mean of the per-user robust difference scales,
converted to an equivalent white-noise sample standard deviation.

The synthetic transform adds channel-wise white Gaussian noise whose variance
is `max(0, sigma_target² - sigma_source²)`. It can make a source noisier. It
cannot make a source quieter. Five draws use seed 0. The classifier is the same
logistic model as the median/IQR experiment. Test users are not used to
estimate scales.

## Scales

Every calibration user-device pair had 13,000–34,000 low-motion differences.
Accelerometer scales are about 0.011–0.027 m/s². Gyroscope scales are about
0.001–0.005 rad/s. `s3mini_1` is the quieter accelerometer, not the noisier
one. The largest noise added to any channel is 0.024 m/s².

## Prediction

On the same 60 cross-device cells:

| Forecast | Degradation MAE | Pearson r | Spearman r | Sign agreement | Mean synthetic degradation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Predict no shift | 0.109 | — | — | — | 0.000 |
| Low-motion white noise | 0.108 | -0.003 | 0.029 | 0.65 | 0.000 |
| Median/IQR affine | 0.122 | 0.323 | 0.440 | 0.85 | 0.130 |

The largest absolute synthetic degradation under the noise model is 0.007.
Mean real degradation on these cells is 0.056. The noise forecast matches the
null forecast. Its sign agreement is computed on changes smaller than 0.008
and does not show a directional effect. The affine model still over-predicts
the size of the drop, and it remains the only one of the two with a positive
association.

Machine-readable result: `results/reviewed/noise_synthetic_baseline.json`.
