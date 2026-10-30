# CPU multi-model gain and ranking audit

**Status:** frozen-protocol exploratory Phase-A result. Protocol commit
`276298a`; implementation commit `0cce670`. This validates the statistical-
feature pipeline and model generality; it is not the final raw-sequence RQ2
experiment.

## Completeness and reproduction

All seven predeclared configurations completed. The artifact contains 825
seed-level rows and 525 seed-aggregated configuration rows. Random forest and
extra trees have exactly three seeds per cell; deterministic configurations
have one. The new `logistic_all` degradation values match the earlier committed
gain experiment to within 5e-7, attributable to earlier six-decimal JSON
rounding.

The full matrix completed on the existing Apple Silicon CPU in approximately
four minutes. No additional CPU, memory, or GPU hardware is required for this
stage.

## Gain prediction by configuration

Subject-macro-F1 gives every test participant equal weight:

| Configuration | Degradation MAE | Pearson r | Spearman r | Slope | R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Extra Trees | **0.0419** | **0.817** | 0.716 | 1.186 | **0.667** |
| Random Forest | 0.0508 | 0.491 | 0.365 | 1.571 | 0.241 |
| Acc-only Logistic | 0.0531 | 0.753 | 0.711 | 0.986 | 0.568 |
| All-sensor Logistic | 0.0695 | 0.677 | 0.663 | 1.015 | 0.458 |
| Linear SVM | 0.0774 | 0.760 | 0.624 | 0.942 | 0.578 |
| RBF SVM | 0.0777 | 0.642 | 0.573 | 1.194 | 0.412 |
| Gyro-only Logistic | 0.0631 | — | — | 0.000 | 0.000 |

The gyroscope-only control has exactly zero synthetic degradation because the
gain transform changes only acceleration and its selected features contain no
acceleration information. Undefined correlation is therefore the expected
manipulation check, not a failed run.

Six acceleration-using configurations show positive real-versus-synthetic
association. The effect is not unique to a linear classifier: Extra Trees has
the strongest agreement, while Random Forest is markedly weaker despite sharing
a tree-ensemble representation. Model inductive bias still matters.

Pooled and subject-macro conclusions are similar. Subject weighting improves
Extra Trees, acceleration-only logistic, random forest, and RBF SVM, so unequal
retained window counts do not manufacture the overall positive result.

## Seven-configuration ranking

| Ranking target | Kendall tau-b | Spearman | Top-2 overlap | Pairwise accuracy |
| --- | ---: | ---: | ---: | ---: |
| Pooled robustness degradation | 0.310 | 0.374 | 0.475 | 0.654 |
| Subject-macro robustness degradation | **0.323** | **0.403** | **0.517** | **0.662** |
| Pooled target performance | 0.583 | 0.708 | 0.658 | 0.791 |
| Subject-macro target performance | **0.660** | **0.776** | **0.725** | **0.830** |

Scalar gain predicts which configuration will perform best on the target more
reliably than it predicts the exact ordering of degradation relative to each
configuration's own source score. This distinction is central: target-
performance ranking is already promising, while robustness-ranking evidence is
only moderate.

## Heterogeneity

Subject-macro performance-ranking Kendall tau-b is positive for every target:
0.778 (`nexus4_1`), 0.675 (`nexus4_2`), 0.651 (`s3_1`), 0.683 (`s3_2`), and
0.516 (`s3mini_1`). It is also positive across folds: 0.581, 0.667, and 0.733.

Robustness ranking is less stable across folds: Kendall tau-b is 0.100, 0.273,
and 0.595 for folds 0–2. The mean positive result must not hide this variation.
Fold 0 includes the largest asymmetric domain shifts and is the hardest setting
for a scalar gain forecast.

## Decision

Phase A succeeds as a pipeline/generalization audit:

- gain agreement extends beyond the original logistic model;
- subject-equal weighting preserves the conclusion;
- synthetic target-performance ranking is meaningfully above chance;
- exact robustness ordering remains only moderately predictable.

The next phase should add raw-sequence neural models under a separately frozen
protocol. The current seven configurations share handcrafted statistical
features and cannot by themselves establish the final RQ2 claim.

Machine-readable artifact:
`results/reviewed/cpu_multimodel_gain_audit.json`.
