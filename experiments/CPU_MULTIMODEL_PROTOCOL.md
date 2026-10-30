# CPU multi-model gain and ranking audit protocol

**Status:** frozen exploratory Phase-A protocol. Commit before implementing or
running the multi-model matrix. This is a pipeline and generality audit, not the
final confirmatory RQ2 experiment with raw-sequence neural models.

## Purpose

Test whether the scalar accelerometer-gain result survives heterogeneous CPU
classifiers and whether synthetic degradation contains useful information about
model/configuration robustness ranking.

## Fixed data and separation

- Use the committed 5-second, 50 Hz anti-aliased windows and 78 statistical
  features.
- Use the fixed subject folds and five primary phone instances.
- Select hyperparameters using the single validation user for the source device.
- Refit on train plus validation users after selection.
- Estimate gain only from both training calibration users, without labels.
- Test users and target-device scores never enter hyperparameter or gain
  selection.

## Fixed configurations

1. six-axis standardized balanced logistic regression;
2. six-axis standardized balanced linear SVM;
3. six-axis standardized balanced RBF SVM;
4. balanced random forest;
5. balanced extremely randomized trees;
6. acceleration-only logistic control, including acceleration magnitude and
   acceleration correlation features;
7. gyroscope-only logistic control, including gyroscope magnitude and
   gyroscope correlation features.

Hyperparameter grids:

- logistic and linear SVM: `C` in {0.1, 1, 10};
- RBF SVM: `C` in {1, 10}, `gamma="scale"`;
- random forest and extra trees: 400 trees, `max_features="sqrt"`,
  `min_samples_leaf` in {1, 5}.

All classifiers use balanced class weighting. Ties in validation macro-F1 are
resolved by the listed grid order, which places the simpler/smaller option
first.

## Seeds and experimental units

Random forest and extra trees use seeds 0, 1, and 2. Deterministic linear,
kernel, and logistic configurations are run once. Seeds are nested repetitions,
not separate model configurations. For ranking, stochastic scores are averaged
across seeds before assigning one rank to the configuration.

## Fixed synthetic condition

Evaluate identity real target data and the committed scalar gain transform only:

```text
acc_fake = acc_source * (gain_target / gain_source)
gyro_fake = gyro_source
```

No bias, noise, timing, spectral, quantization, or clipping factor is included.
This isolates whether the strongest single factor generalizes across models.

## Scores

Retain two evaluation summaries:

1. pooled macro-F1 over all test windows, matching prior results;
2. subject-macro-F1: compute macro-F1 separately for each of the three test
   users and average them equally.

The second score audits unequal device-specific window retention. It does not
make source and target windows mechanically paired, so remaining coverage bias
must still be reported.

## Degradation prediction outcomes

For every configuration, report the same 60 off-diagonal cells and summarize
gain-predicted versus real degradation using MAE, RMSE, Pearson, Spearman,
calibration slope/intercept, and R-squared. Report pooled and subject-macro
scores separately. Stochastic summaries first average the three seed-level
degradations within each cell.

## Ranking outcomes

For each fold/source/target cell, rank the seven configurations after seed
aggregation.

- Primary robustness ranking: lower real degradation versus lower synthetic
  degradation.
- Secondary performance ranking: higher real target score versus higher
  synthetic target score.

Report Kendall tau-b, Spearman correlation, top-2 overlap, and pairwise ordering
accuracy per cell and as a mean across 60 cells. Ties use the metric's standard
tie handling and count as half-correct in pairwise ordering accuracy when only
one side is tied.

## Interpretation boundary

- Consistent gain prediction across classifier families supports model
  generality for statistical features.
- Ranking agreement above chance supports proceeding to full RQ2, but is not
  confirmatory evidence because five configurations share the same handcrafted
  representation and the dataset has already been explored.
- Final RQ2 requires raw-sequence models such as compact 1D CNN,
  DeepConvLSTM/ResNet1D, and carefully justified pretrained IMU encoders.
- No CPU model or hyperparameter will be added or removed after inspecting its
  target ranking unless the run fails a pre-target implementation test; all
  failures remain documented.

## Compute policy

This Phase-A matrix is CPU-only. The current 21,305-by-78 feature table is small
enough for the existing machine. No hardware purchase is justified by this
phase. GPU rental is considered only after the raw-sequence protocol is frozen.
