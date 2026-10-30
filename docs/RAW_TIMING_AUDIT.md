# HHAR raw sampling and timestamp audit

**Status:** exploratory acquisition audit. Protocol commit `8cb45b5`;
implementation commit `fdbf670`. Only the two training calibration users per
fold are profiled. Signal values, activity labels, validation users, and test
users are not read.

## Device timing profiles

The main device distinction is stable across folds and calibration users:

| Device/sensor pattern | Median native rate | Normalized interval MAD | Gaps >100 ms/min | Gap fraction |
| --- | ---: | ---: | ---: | ---: |
| Nexus 4 accelerometer/gyro | about 198.6 Hz | about 0.003–0.006 | 2.6–5.6 | about 0.0005 |
| Galaxy S3 accelerometer | about 98.8 Hz | about 0.001–0.003 | 3.3–4.6 | about 0.0007 |
| Galaxy S3 gyroscope | about 200 Hz | about 0.004–0.006 | 3.5–5.3 | about 0.0004 |
| `s3mini_1` accelerometer | about 99.9 Hz | 0.024–0.027 | 22.1–27.1 | 0.0038–0.0047 |
| `s3mini_1` gyroscope | about 99.9 Hz | 0.027–0.034 | 22.2–27.0 | 0.0038–0.0047 |

For `s3mini_1`, individual calibration-user gap rates range from 20.4 to 27.9
per minute. Its high fragmentation is therefore not an artifact of a single
calibration user. This independently agrees with the 9,717 legal resampled
segments reported in the preprocessing audit.

## Association with gain residuals

The strongest fixed directional factors are:

| Target minus source factor | Signed-residual Pearson | Spearman |
| --- | ---: | ---: |
| Accelerometer normalized interval MAD | 0.641 | 0.528 |
| Gyroscope normalized interval MAD | 0.671 | 0.532 |
| Accelerometer gaps per minute | 0.668 | 0.443 |
| Gyroscope gaps per minute | 0.671 | 0.457 |

The direction is physically coherent: moving toward the more irregular device
is associated with a more positive residual, meaning scalar gain under-predicts
the real degradation. Moving away from it is associated with gain
over-predicting degradation or with negative transfer degradation.

Fold-specific Pearson correlations for target-minus-source gyroscope interval
MAD are 0.888, 0.507, and 0.566. The corresponding gap-rate correlations are
0.890, 0.436, and 0.578. The direction therefore repeats in all three folds.

## Localization check and decision

The association is device-specific. After removing every transfer involving
`s3mini_1`:

- gyroscope interval-MAD Pearson changes from 0.671 to -0.003;
- accelerometer interval-MAD Pearson changes from 0.641 to -0.051;
- gyroscope gap-rate Pearson changes from 0.671 to -0.268;
- accelerometer gap-rate Pearson changes from 0.668 to -0.342.

This post-hoc localization check shows that timing irregularity identifies the
`s3mini_1` domain axis; it does not explain residual variation among the other
four devices. Under the frozen decision rule, it therefore does not justify a
general sampling-jitter perturbation.

The result remains valuable as a severe-device case study. A future fixed
experiment may simulate `s3mini_1`-like gaps and interval jitter, but it must be
reported separately from generic cross-device claims. The main factor search
now proceeds to quantization, clipping, and filter-response characterization.

Machine-readable artifact: `results/reviewed/raw_timing_audit.json`.
