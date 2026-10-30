# Low-motion spectral-shape audit

**Status:** exploratory mixed/negative audit. Protocol commit `25da760`;
implementation commit `174b8f7`. No filter was fit and no target score was used
to construct a spectral profile.

## Data sufficiency

All 30 fold/device/sensor profiles satisfy the requirement that both calibration
users contribute at least 10 non-overlapping 128-sample low-motion windows.
Individual profile counts range from 34 to 216 windows. Windows remain inside a
single continuous low-motion run and preprocessing segment.

## Residual associations

The strongest directional associations are consistent with a smoother target
signal producing a more positive gain residual:

| Target minus source feature | Signed-residual Pearson | Spearman |
| --- | ---: | ---: |
| Accelerometer 10–18 Hz power fraction | -0.579 | -0.570 |
| Accelerometer spectral centroid | -0.512 | -0.514 |
| Accelerometer high/mid log power | -0.484 | -0.455 |
| Accelerometer lag-one autocorrelation | 0.548 | 0.582 |
| Gyroscope 5–10 Hz power fraction | -0.522 | -0.360 |

Accelerometer high-frequency fraction has fold-specific Pearson values -0.768,
-0.384, and -0.485. Accelerometer lag-one autocorrelation gives 0.742, 0.439,
and 0.523. The direction therefore repeats across folds.

After removing every `s3mini_1` transfer, accelerometer lag-one autocorrelation
retains Pearson 0.332 and Spearman 0.373. High-frequency fraction weakens to
-0.234 and -0.295. The spectral signal is not entirely identical to the severe
timing-device indicator, but much of its magnitude is associated with that
domain axis.

## Calibration-user stability failure

The protocol also requires stable device ordering between the two calibration
users. That condition is not met strongly enough to fit a device filter:

| Feature | Fold 0 user rank agreement | Fold 1 | Fold 2 |
| --- | ---: | ---: | ---: |
| Acceleration 10–18 Hz fraction | 0.30 | 0.30 | 0.60 |
| Acceleration spectral centroid | 0.40 | 0.00 | 0.50 |
| Acceleration lag-one autocorrelation | 0.40 | 0.70 | 0.50 |
| Gyroscope 10–18 Hz fraction | 0.00 | 1.00 | 0.90 |

Low-motion recordings still contain participant-specific micro-motion, posture,
and device interaction. Without a shared mechanical input or reliable
cross-phone synchronization, these aggregate spectra cannot be identified as
manufacturer transfer functions.

## Decision

No filter or colored-noise transform will be fit from this audit. Doing so would
convert participant-dependent spectral differences into an outcome-guided
device model. The repeated directional association remains a hypothesis for a
future controlled dataset or synchronized hardware experiment.

The next calibration-only audit should use quantization and clipping signatures,
which can be measured from raw value representation without requiring the same
mechanical input across devices.

Machine-readable artifact:
`results/reviewed/low_motion_spectral_audit.json`.
