# Post-target fold heterogeneity diagnostic

**Status:** post-target diagnostic. This analysis was specified after the
frozen Phase-B result and is not presented as a preregistered confirmation.

## Why Fold 2 matters

The mean subject-macro performance-ranking increment over the unchanged source
ranking is 0.152 Kendall tau-b, but the fold increments are 0.029, 0.091, and
0.336. This audit checks whether the third value is explained by unequal
device-pair coverage, weak source models, compressed source scores, or a more
extreme calibration transform.

## Fold composition and coverage

| Fold | Test users | Unlabelled calibration users | Directed device cells |
| --- | --- | --- | ---: |
| 0 | a, c, e | b, g | 20 |
| 1 | b, f, i | a, h | 20 |
| 2 | d, g, h | c, i | 20 |

Every fold contains the same five sources, five targets, and 20 off-diagonal
source-to-target directions. Unequal device coverage does not explain Fold 2.

## Source-baseline weakness

| Fold | Source-ranking tau-b | Synthetic-ranking tau-b | Increment |
| --- | ---: | ---: | ---: |
| 0 | 0.544 | 0.573 | 0.029 |
| 1 | 0.638 | 0.729 | 0.091 |
| 2 | 0.393 | 0.729 | 0.336 |

Fold 2's increment is large primarily because its unchanged source ranking
transfers poorly. That weak ranking is not caused by uniformly weak or nearly
tied source scores:

| Fold | Mean source subject-macro F1 | Mean within-source configuration SD | Mean within-source range |
| --- | ---: | ---: | ---: |
| 0 | 0.642 | 0.089 | 0.276 |
| 1 | 0.657 | 0.070 | 0.247 |
| 2 | 0.770 | 0.071 | 0.235 |

Fold 2 has the highest mean source performance, and its model-score dispersion
is comparable to Fold 1. Its low baseline therefore reflects a real ordering
change between source and target scores for test users d, g, and h rather than
a low-score or compressed-range artifact.

The fitted acceleration gains also have similar device ranges across folds
(approximately 0.071, 0.078, and 0.074). Fold 2 is not receiving an obviously
larger synthetic transform.

## Leave-one-fold-out sensitivity

| Excluded fold | Remaining subject-macro delta tau-b | Remaining pairwise delta |
| --- | ---: | ---: |
| 0 | 0.213 | 0.107 |
| 1 | 0.182 | 0.091 |
| 2 | **0.060** | **0.030** |

The incremental signal remains positive without Fold 2, but it becomes small.
This is the most defensible sensitivity result for the current three-fold
design.

## Fold-cluster bootstrap

A deterministic 10,000-replicate bootstrap resampling the three folds as
clusters gives a descriptive percentile interval of [0.029, 0.336] around the
subject-macro delta tau-b estimate. This interval is highly discrete and must
not be treated as strong inferential evidence: only three clusters exist, and
all possible resamples are combinations of the same three observed fold means.

## Conclusion and unresolved cause

The result is not a device-coverage artifact, a low-source-performance artifact,
a compressed-score artifact, or an obviously stronger-gain artifact. The
remaining explanation supported by current outputs is a participant-fold by
device interaction: for test users d, g, and h, source model quality transfers
unusually poorly while gain-adjusted ranking transfers well.

The reviewed artifact stores the equal-subject mean, not separate per-user
scores. Distinguishing d, g, and h therefore requires a new per-subject
inference pass over the frozen checkpoints. Until that diagnostic or an
external dataset is complete, the result should be described as preliminary
incremental evidence with strong fold heterogeneity.
