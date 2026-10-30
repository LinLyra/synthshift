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

