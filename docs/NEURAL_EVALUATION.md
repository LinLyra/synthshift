# Frozen neural evaluation and aggregation

Target evaluation is a gated stage. It cannot begin until all 135 entries in
`experiments/neural_job_manifest.json` have a complete formal checkpoint with a
matching job identity, protocol commit, and SHA-256 checksum.

## Per-checkpoint evaluation

Each final refit model is evaluated on the frozen test users for:

1. untouched windows from its source device;
2. real windows from each of the five primary target devices;
3. source windows transformed by the frozen target/source acceleration-gain
   ratio for each of the five targets.

Gain is applied to acceleration in physical units before the checkpoint's
source normalization. Gyroscope channels are unchanged. Both pooled macro-F1
and the equal-subject-weighted mean of subject-level macro-F1 are recorded.
Every evaluation row carries the checkpoint hash that produced it; a cached row
is rejected if that checkpoint changes.

Per-job evaluation files are written atomically under `runs/neural/evaluation`
so an interrupted evaluation can resume without recomputing valid completed
jobs. These operational files remain ignored by Git.

## Frozen aggregation

The expected result cardinalities are checked before analysis:

- 675 seed-level rows: 135 checkpoints x 5 targets;
- 225 neural aggregate rows: 3 folds x 5 sources x 5 targets x 3 architectures;
- exactly seeds 0, 1, and 2 in every neural aggregate;
- 60 off-diagonal fold/source/target cells;
- 10 configurations in every combined ranking cell: the immutable seven CPU
  configurations plus three neural architectures.

The final artifact reports per-architecture synthetic-to-real fits and the
predeclared ranking metrics: Kendall tau-b, Spearman correlation, top-2 overlap,
and pairwise ordering accuracy for robustness and target performance, under
both pooled and subject-macro scoring. It records the SHA-256 of the immutable
CPU artifact used in the combined analysis.

No result-dependent architecture, optimization, metric, or aggregation choice
is made in this stage.
