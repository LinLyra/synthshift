# Frozen raw-sequence neural results

**Status:** completed Phase-B evaluation under the protocol frozen at commit
`b564116`. This is the first target evaluation of the raw-sequence neural grid.

## Execution and integrity

The complete matrix contains 135 source-training jobs: three subject folds,
five source phones, three architectures, and three seeds. All 135 checkpoints
passed the identity, protocol-commit, and SHA-256 gate before any target score
was computed. Evaluation produced the prespecified cardinalities:

- 675 seed-level rows (135 checkpoints x five targets);
- 225 seed-aggregated neural rows;
- 60 off-diagonal fold/source/target cells;
- ten configurations per ranking cell (seven frozen CPU configurations and
  three neural architectures).

The reviewed artifact is
`results/reviewed/neural_combined_ranking_audit.json` (SHA-256
`59ac4adcbbe28f0e41cab3920222d2c9b80db3159c1b167d36785bba9681229d`). It
records the SHA-256 of the immutable CPU input and the checkpoint hash behind
every seed-level neural result.

## Combined model-ranking result

The table averages the prespecified ranking metric over 60 off-diagonal cells.
Neural seeds are averaged within architecture before ranking, so seeds are not
treated as independent model configurations.

| Score and ranking target | Kendall tau-b | Spearman | Top-2 overlap | Pairwise accuracy |
| --- | ---: | ---: | ---: | ---: |
| Pooled robustness | 0.344 | 0.425 | 0.450 | 0.672 |
| Pooled target performance | 0.627 | 0.749 | 0.608 | 0.814 |
| Subject-macro robustness | 0.355 | 0.434 | 0.450 | 0.678 |
| Subject-macro target performance | 0.677 | 0.794 | 0.642 | 0.839 |

The calibrated synthetic target is substantially more informative about which
model will perform best on a real target than about which model will lose the
least performance relative to its own source score. This distinction matters:
target performance combines source-model quality and transfer, whereas
robustness isolates the change from each model's source baseline.

## Fold heterogeneity

Subject-macro target-performance agreement is positive in every fold, while
robustness agreement varies materially:

| Fold | Robustness tau-b | Robustness pairwise | Performance tau-b | Performance pairwise |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.173 | 0.587 | 0.573 | 0.787 |
| 1 | 0.293 | 0.648 | 0.729 | 0.864 |
| 2 | 0.598 | 0.799 | 0.729 | 0.864 |

The mean ranking result must therefore not be interpreted as uniform behavior
across participant folds.

## Neural degradation prediction

These fits use the 60 off-diagonal cells for each seed-aggregated architecture.

| Architecture | Score | Degradation MAE | Pearson r | Spearman r | Mean synthetic degradation |
| --- | --- | ---: | ---: | ---: | ---: |
| Compact CNN | pooled | 0.076 | 0.297 | 0.377 | 0.008 |
| Compact CNN | subject-macro | 0.082 | 0.274 | 0.375 | 0.007 |
| DeepConvLSTM | pooled | 0.049 | 0.665 | 0.701 | 0.018 |
| DeepConvLSTM | subject-macro | 0.051 | 0.659 | 0.708 | 0.019 |
| ResNet1D | pooled | 0.082 | 0.390 | 0.382 | 0.017 |
| ResNet1D | subject-macro | 0.090 | 0.344 | 0.384 | 0.016 |

The scalar gain generalizes best to DeepConvLSTM among the three neural
families. It under-predicts the mean subject-macro real degradation for
DeepConvLSTM (0.029 real versus 0.019 synthetic) and ResNet1D (0.025 versus
0.016). Compact CNN has a small mean real degradation (0.006), but weak
cell-level association. Architecture behavior is therefore not interchangeable.

## Interpretation boundary

This result supports a limited conclusion: one calibration statistic can carry
useful information about cross-device target-performance ordering and some
architecture-specific degradation, but it does not reproduce the complete
robustness ordering. The frozen protocol reports descriptive agreement across
the 60 dependent cells; the draft project preregistration did not finalize a
cluster-bootstrap confidence-interval procedure before this run. No
independence-based p-value or post-hoc significance claim is made.

The study is also limited to HHAR waist-phone HAR. It does not establish
generalization to watches, pose estimation, odometry, another dataset, or an
IMU foundation model. Those are external-validation extensions rather than
retroactive additions to this frozen result.

## Post-target source-ranking baseline

After the frozen result was inspected, we added the most important trivial
baseline as an explicitly post-target diagnostic: use each configuration's
unchanged source-device performance ranking to predict its real target-device
performance ranking. This tests whether synthetic calibration adds information
beyond the fact that stronger source models often remain stronger targets.

| Score | Predictor | Kendall tau-b | Spearman | Top-2 overlap | Pairwise accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Pooled | Source performance | 0.510 | 0.626 | 0.633 | 0.755 |
| Pooled | Synthetic target | 0.627 | 0.749 | 0.608 | 0.814 |
| Subject-macro | Source performance | 0.525 | 0.647 | 0.583 | 0.763 |
| Subject-macro | Synthetic target | 0.677 | 0.794 | 0.642 | 0.839 |

For subject-macro scoring, synthetic calibration improves mean tau-b by 0.152
and pairwise accuracy by 0.076. Tau-b improves in 37 of 60 cells, ties in 7,
and worsens in 16. The corresponding pooled improvement is 0.118, although
pooled top-2 overlap decreases by 0.025.

The incremental subject-macro tau-b is heterogeneous by participant fold:
0.029, 0.091, and 0.336 for folds 0, 1, and 2. Thus synthetic calibration does
beat the trivial source-ranking baseline on average, but most of the mean gain
comes from fold 2. The defensible conclusion is incremental signal, not yet
fold-stable incremental validity.

Leaving out Fold 2 reduces the subject-macro increment to 0.060 (pairwise
accuracy increment 0.030). A three-cluster bootstrap gives a descriptive
percentile interval [0.029, 0.336], but three folds are too few for that interval
to support a strong inferential claim. The detailed audit in
[`docs/FOLD_DIAGNOSTIC.md`](FOLD_DIAGNOSTIC.md) shows that Fold 2 has equal
device-pair coverage, the highest mean source performance, comparable
configuration-score dispersion, and a similar gain range. Its large increment
is driven by unusually poor transfer of the source ranking for test users d,
g, and h, not by a trivial coverage or low-score artifact.

This analysis is stored separately in
`results/reviewed/performance_ranking_baseline_audit.json`; it does not rewrite
the frozen Phase-B artifact or upgrade a post-target diagnostic into a
preregistered outcome.
