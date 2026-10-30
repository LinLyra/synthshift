# Exploratory CPU baseline

**Status:** engineering and feasibility baseline, not a confirmatory test of the
SynthShift hypotheses.

## Protocol

- Input: five-second, anti-aliased 50 Hz, six-axis windows with 50% overlap.
- Features: 78 deterministic distribution, magnitude, temporal-difference, and
  within-sensor correlation features.
- Classifier: standardized class-balanced multinomial logistic regression.
- Hyperparameter: `C` selected from 0.1, 1, and 10 using the fixed validation
  subject for each source-device/fold combination.
- Evaluation: train on one physical source device and score every target device
  on subject-disjoint outer-test users; three outer folds.
- Metric: macro-F1, averaged across folds.

## Real cross-device matrix

Rows are training devices and columns are evaluation devices.

| Source \ Target | nexus4_1 | nexus4_2 | s3_1 | s3_2 | s3mini_1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nexus4_1` | 0.777 | 0.731 | 0.708 | 0.592 | 0.593 |
| `nexus4_2` | 0.773 | 0.793 | 0.772 | 0.713 | 0.686 |
| `s3_1` | 0.712 | 0.726 | 0.746 | 0.677 | 0.647 |
| `s3_2` | 0.615 | 0.768 | 0.744 | 0.776 | 0.684 |
| `s3mini_1` | 0.532 | 0.666 | 0.653 | 0.724 | 0.618 |

Mean same-device macro-F1 is 0.742; mean off-diagonal cross-device macro-F1 is
0.686, an absolute reduction of 0.056. The effect is strongly asymmetric. For
example, `nexus4_1` loses 0.185 when evaluated on `s3_2`, while a model trained
on `s3mini_1` performs better on several other devices than on its own device.

This is exactly why the project must predict both degradation magnitude and
model/device ranking rather than assume every perturbation causes monotonic
decline. The matrix establishes that a measurable real-device phenomenon exists
and that a single average domain-gap number would hide important structure.

## Interpretation boundaries

- This baseline does not yet contain synthetic shifts, so it cannot answer RQ1
  or RQ2.
- Overlapping windows are never treated as independent inferential units;
  uncertainty will be bootstrapped at participant/device-pair level.
- The classifier represents one model family and one seed. It is a data-pipeline
  validation baseline, not the final model-ranking population.
- Phone placement was nominally shared, but physical position and orientation
  inside the pouch may still contribute to the observed device matrix.

Machine-readable predictions, metrics, class order, confusion matrices, and
selected hyperparameters are stored in
`results/reviewed/statistical_baseline_antialias.json`. The original
pre-filter result is retained as a preprocessing sensitivity artifact.
