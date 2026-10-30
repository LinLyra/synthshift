# SynthShift preregistration (draft)

**Status:** Draft. This document must be finalized and committed before the
confirmatory model comparison is run.

## Research question

Can a synthetic target device, calibrated from measurable device properties,
predict the failure magnitude and robustness ranking observed on a real target
device in inertial human-activity recognition?

## Scope

- Dataset: HHAR activity-recognition recordings.
- Primary modality: phone accelerometer and gyroscope.
- Primary comparisons: phone-to-phone only, with phones held at the same nominal
  waist placement during collection.
- Excluded from primary claims: phone-to-watch comparisons and any newly
  collected participant data.

## Research questions

### RQ1 — degradation prediction

How accurately does calibrated synthetic shift predict real cross-device score
degradation, compared with uncalibrated generic perturbations?

### RQ2 — model ranking

Does the model ranking under synthetic target-device data agree with the ranking
under the real target device?

### RQ3 — explanatory factors

When synthetic and real outcomes disagree, how much is attributable to sampling
rate/jitter, bias/scale, noise, filtering, or residual mounting/orientation?

## Experimental units

- Device instance and device model are distinct hierarchy levels.
- Random seeds are repeated measurements nested within a model configuration.
- A model family is not multiplied by treating its random seeds as independent
  methods.

## Planned comparisons

The initial model set will include heterogeneous families, subject to successful
reproduction: DeepConvLSTM, a compact 1D CNN, a residual 1D network,
InrtlWrist frozen, InrtlWrist fine-tuned, LIMU-BERT, a randomly initialized
matched encoder, and sensor-subset controls.

## Data separation

1. Model training, calibration, validation, and final evaluation partitions must
   be disjoint according to the finalized split protocol.
2. Target-device calibration may use unlabelled target data only in the
   explicitly named calibration setting.
3. No parameter may be estimated from the final target evaluation partition.
4. All split manifests and hashes will be versioned.

The fixed subject folds are defined in
`configs/splits/hhar_subject_folds.json`. Every subject is tested exactly once
across three outer folds. Each fold contains five training subjects, one
validation subject, and three test subjects. S1 target calibration users are a
strict subset of training subjects and their activity labels remain unavailable
to the calibration method.

## Preprocessing decisions supported by the raw-data audit

- Primary six-axis candidate devices: `nexus4_1`, `nexus4_2`, `s3_1`, `s3_2`,
  and `s3mini_1`.
- `s3mini_2` is exploratory because of limited duration and atypical timing;
  Galaxy S+ is accelerometer-only because gyroscope data are unavailable.
- Each sensor is segmented before interpolation, then resampled independently
  to a common 50 Hz grid.
- No window may cross a forward gap over 100 ms, a backward clock jump over one
  second, an activity transition, or a missing-label interval. Smaller negative
  deltas are treated as local delivery reorder and stably sorted within segment.
- A 10 ms nearest-neighbour match is an alignment diagnostic only and is not
  used as the training-time fusion method.

## Two deployment settings

### S0 — target-free

No target-device recording is used to configure the synthetic transformation.

### S1 — unlabelled device calibration

A small, disjoint, unlabelled calibration partition may be used to estimate
device properties. Final evaluation remains held out.

## Outcomes

- Performance degradation error: MAE, RMSE, calibration slope/intercept, and
  coefficient of determination where appropriate.
- Ranking agreement: Kendall tau-b, Spearman correlation, top-k overlap, and
  pairwise ordering accuracy.
- Uncertainty: participant/device-pair bootstrap intervals with seeds retained as
  nested repetitions.
- Diagnostic outcomes: class-level errors and corruption-factor ablations.

No monotonic performance-degradation assumption is imposed. Invariance to a
shift is a valid observation.

## Decision policy

The study proceeds if HHAR metadata permits credible phone-to-phone comparisons
and at least one measurable cross-device behavior exists. The study may proceed
whether calibrated synthesis agrees, partially agrees, or disagrees with real
device behavior; all three outcomes answer the research question.

The study stops or is redesigned if device identity, timing, placement, or data
separation cannot support the claimed comparison, or if the literature review
finds the calibrated synthetic-to-real prediction study has already been
performed at equivalent scope.

## Items to finalize before confirmation

- Device-pair inclusion rules and minimum sample counts.
- Primary score and confidence-interval procedure.
- Calibration-budget sizes.
- Model hyperparameter-selection protocol.
- Criteria distinguishing exploratory from confirmatory analyses.

