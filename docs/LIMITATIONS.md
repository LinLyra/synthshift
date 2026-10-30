# Limitations register

This document is updated throughout the project.

Initial known limitations:

- HHAR contains only nine participants.
- A tight waist pouch reduces but may not eliminate orientation and mounting
  differences between phones.
- Target-device calibration can become transductive if calibration and evaluation
  recordings are not separated rigorously.
- A single dataset limits external validity.
- Synthetic agreement on HAR does not establish agreement for pose, odometry, or
  clinical tasks.
- A single constant Arrival_Time offset does not yet support paired-motion
  sensor maps. Most device pairs fail the pre-set half-session or cross-user
  stability gate, and `s3mini_1` has no stable pair.
- HHAR still-experiment folders do not identify the activity-study phone
  instances. Low-motion white-noise differences among those instances are too
  small to account for the observed cross-device score drop.
- A scalar accelerometer gain leaves gyroscope scale, axis bias, and most of
  the same-model instance shift unaccounted for. The current positive gain
  result uses one linear model.
- Adding a low-motion median gyroscope bias gives only a small overall
  improvement whose 10-device-pair cluster-bootstrap interval includes zero.
  Calibration-user disagreement reaches 0.0347 rad/s on one `s3_2` axis, and
  the same-model improvement is concentrated in Galaxy S3 rather than Nexus 4.
- Raw interval jitter and >100 ms gap rate strongly identify the directional
  `s3mini_1` residual in all folds, but their association disappears when that
  device is removed. They support a severe-device case study, not a general
  timing explanation across the remaining phones.
- Low-motion spectral shape is directionally associated with gain residuals,
  but calibration-user device rankings are unstable in several folds. Without
  a shared mechanical input, these spectra cannot identify hardware filters.
- Exported quantization steps are stable device fingerprints but do not track
  gain residuals. An apparent ±19.613 m/s² clipping rail is specific to
  `s3mini_1`; clipping cannot be inverted for transfers away from that device.
- Applying the fixed `s3mini_1` ±2g rail after gain changes only 0.0186% of
  scalar values and improves none of 12 valid target-direction cells for the
  linear model. The rail is a domain marker, not a supported causal factor.
- Seven CPU configurations support gain generality for handcrafted features,
  but robustness-ranking Kendall tau-b is only 0.323 under subject-equal
  scoring and varies from 0.100 to 0.595 across folds. Raw-sequence neural
  models are still required for the final model-ranking claim.

