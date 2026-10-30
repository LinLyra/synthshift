# Subject-disjoint evaluation folds

The outer evaluation uses three fixed folds. Every participant appears in the
test set exactly once, and no participant appears in more than one partition of
the same fold.

| Fold | Train | Validation | Test | Unlabelled target calibration |
| --- | --- | --- | --- | --- |
| `fold_0` | b, d, f, g, i | h | a, c, e | b, g |
| `fold_1` | a, d, e, g, h | c | b, f, i | a, h |
| `fold_2` | a, c, e, f, i | b | d, g, h | c, i |

The fold-2 validation subject was changed from f to b before any baseline result
was produced because f has no `s3_1` accelerometer stream. Test membership was
not changed. The folds have 6,980–7,119 primary-device test windows under the
provisional
five-second window definition. Every train, validation, and test partition has
nonzero coverage for all six activities.

The S1 calibration users are a strict subset of training users. Calibration may
read target-device signals but not activity labels. Test-user target data are
never used to estimate device characteristics, normalization statistics,
corruption parameters, thresholds, or hyperparameters.

The split file is validated automatically for partition overlap, complete user
coverage, calibration containment, and exactly-once outer testing. Its SHA-256
is recorded in `docs/audits/hhar_split_coverage.json`.

Canonical configuration: `configs/splits/hhar_subject_folds.json`.
