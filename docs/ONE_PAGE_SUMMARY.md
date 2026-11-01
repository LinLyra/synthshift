# SynthShift: Can calibrated synthetic IMU shifts predict real device failures?

## Research summary

Wearable and mobile sensing models often degrade when deployed on a different
physical device, even when sensor modality, nominal placement, and activity
classes remain unchanged. Synthetic corruptions are widely used to evaluate
robustness, but a basic validity question remains: **does a synthetic target
device predict how models actually behave on the corresponding real device?**

SynthShift studies this question on the HHAR human-activity-recognition dataset.
The primary analysis uses five waist-phone instances—two Nexus 4 devices, two
Samsung Galaxy S III devices, and one Galaxy S III mini—and excludes watches
because their body placement differs. Nine participants are partitioned into
three subject-disjoint folds. Target calibration uses only unlabelled data from
training participants; target test participants and labels are excluded from
gain estimation, normalization, model selection, and training.

The synthetic transformation is deliberately simple. A rotation-invariant
scalar accelerometer gain is estimated from low-gyroscope segments for each
phone, then the source acceleration is multiplied by the target/source gain
ratio. Gyroscope channels remain unchanged. The final ranking comparison covers
ten heterogeneous configurations: seven fixed classical models and three
raw-sequence neural architectures (Compact CNN, DeepConvLSTM, and ResNet1D).
Neural models use three seeds, aggregated within architecture rather than
treated as independent methods. The neural stage comprises 135 formally gated
training jobs and 60 off-diagonal fold/source/target evaluation cells.

The synthetic target predicts real subject-macro target-performance ranking
with mean Kendall tau-b 0.677 and pairwise ordering accuracy 0.839. However,
strong source models may remain strong after transfer even without any device
simulation. We therefore added an explicitly **post-target diagnostic** that
uses unchanged source performance to rank real target performance. This
trivial baseline reaches tau-b 0.525 and pairwise accuracy 0.763. Synthetic
calibration adds **delta tau-b +0.152** and **pairwise accuracy +0.076** on
average, improving tau-b in 37 of 60 cells, tying in 7, and worsening it in 16.

The increment is not stable across participant folds. Fold-specific delta
tau-b values are +0.029, +0.091, and +0.336. Leaving out the high-increment
fold reduces the estimate to **+0.060**. A fold-cluster bootstrap produces a
descriptive interval [0.029, 0.336], but only three clusters exist, so this is
not treated as strong inferential evidence. The high-increment fold has equal
device-pair coverage, the highest mean source performance, comparable
configuration-score dispersion, and a similar calibration-gain range. Its
behavior is therefore not explained by missing device directions, uniformly
weak models, compressed source scores, or a larger perturbation.

The evidence supports a deliberately limited conclusion: **calibrated
accelerometer gain contains preliminary incremental information beyond model
strength alone, but cross-fold stability has not been established.** The study
is limited to one dataset, nine participants, three folds, phone-based HAR, and
relatively small model families. The source-ranking baseline and fold
sensitivity were added after the frozen target result and are labelled as such.
Per-participant attribution, a second multi-device dataset, and IMU foundation
models are future validation steps rather than part of the current claim.

The repository retains null results, fixed split manifests, frozen protocols,
checkpoint hashes, deterministic aggregation, tests for data and evaluation
invariants, and compact reviewed artifacts. The contribution is therefore not
only a positive number, but an auditable example of testing whether a synthetic
sensor benchmark provides information beyond a strong trivial baseline.
