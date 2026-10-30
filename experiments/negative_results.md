# Negative and null results

This log preserves well-formed experiments that fail to support the current
hypothesis. Each entry must link to an experiment ID, configuration, code commit,
and retained summary artifact.

## s1-affine-median-iqr-v0

- Status: exploratory partial failure.
- Commit: `5f47a63`.
- Config: `configs/splits/hhar_subject_folds.json`.
- Artifact: `results/reviewed/affine_synthetic_baseline.json`.
- Method: unlabelled per-channel median/IQR affine map. Each fold uses its two
  training-set calibration users. The map is applied to source test windows.
  Target test users and activity labels are not used to fit the map.

On 60 source-to-target evaluations the degradation MAE is 0.122 and RMSE is
0.187. Pearson r is 0.323 and Spearman r is 0.440. Mean real degradation is
0.056; mean synthetic degradation is 0.130. The sign of the degradation agrees
on 85% of cells. The same-device diagonal is an exact identity, so the gap is
not a feature-extraction mismatch.

The error is concentrated. Sixteen cells whose accelerometer x scale is at
least 1.5 or at most 0.67 have degradation MAE 0.289. The other 44 cells have
MAE 0.062, sign agreement 0.886, and mean synthetic degradation 0.052 against
mean real degradation 0.043. The large scales are accelerometer x into or out
of `s3mini_1`, reaching 2.483 for fold 1 `nexus4_1` to `s3mini_1`. Gyroscope
scales on those same cells stay near 1. Fold 0 has no accelerometer scale
outside 0.70–1.40.

The marginal distribution of a calibration user's whole recording therefore
tracks real degradation when the axis scales are already close, and it
over-predicts degradation when that user's posture or retained-segment mix
differs across phones. Kendall tau 0.627 ranks target devices for this one
linear model, including the same-device cell. It is not the preregistered
model-ranking correlation.

The per-row field `degradation_error` stores synthetic macro-F1 minus real
macro-F1. The summary MAE is computed from the degradation values, not from
that field.

## sync-arrival-magnitude-v0

- Status: gate failed for a constant cross-device clock offset.
- Commit: `1d823be`.
- Config: `configs/splits/hhar_subject_folds.json`.
- Artifact: `docs/audits/hhar_phone_sync.json`.
- Report: `docs/SYNC_AUDIT.md`.

The stability rule was fixed before reading the HHAR result. A calibration
pair is usable only when both users have peak correlation at least 0.30, peak
margin at least 0.10, half-session offset difference at most 50 ms, and the
two users' offsets differ by at most 50 ms. Three of 30 fold-pairs pass, all
in fold 1. No 3×3 sensor map was fit.

## s1-noise-low-motion-v0

- Status: exploratory null. The synthetic forecast matches predicting no shift.
- Commit: `9058b1c`.
- Config: `configs/splits/hhar_subject_folds.json`.
- Artifact: `results/reviewed/noise_synthetic_baseline.json`.
- Report: `docs/NOISE_BASELINE.md`.

Still-experiment files were not used, because their device names do not
identify the activity instances. Low-motion white noise estimated from
calibration users changes macro-F1 by at most 0.007. Degradation MAE is 0.108,
against 0.109 for a forecast of no shift. Pearson r is -0.003.

## s3mini-clipping-case-v0

- Status: frozen-protocol exploratory null.
- Protocol: `417f81f`.
- Implementation: `345a461`.
- Artifact: `results/reviewed/s3mini_clipping_case_study.json`.
- Report: `docs/S3MINI_CLIPPING_CASE_STUDY.md`.

A fixed ±2g clip after acceleration gain was evaluated only for the 12 valid
directed cells whose target is `s3mini_1`. It changes 0.0186% of scalar values
and 2.58% of windows on average. Nine cells are score-identical, three worsen,
and none improve. MAE changes from 0.07903 to 0.07936. The rail is a device
fingerprint but does not explain target degradation for the current model.

