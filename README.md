# SynthShift

**Can calibrated synthetic IMU shifts predict real cross-device model failures
and model rankings?**

SynthShift is a reproducible inertial-sensing research project using the HHAR
multi-device dataset. It calibrates synthetic device shifts from unlabelled
sensor statistics, then tests whether those proxies predict degradation on real
held-out phones.

The primary study uses phone-to-phone comparisons only. HHAR watches are not
used for device-only conclusions because their body placement differs from the
phones. No new human-participant data is required.

## Current status

The data pipeline, subject-disjoint protocol, exploratory factor analysis,
seven-configuration CPU model audit, raw-sequence neural architectures,
resumable training system, and gated evaluation pipeline are implemented. The
frozen 135-job raw-sequence neural grid and target evaluation are complete.

The strongest exploratory factor is a rotation-invariant scalar accelerometer
gain estimated from low-gyroscope segments. Across 60 fold/source/target cells,
it reduced degradation-prediction MAE from approximately 0.109 for a no-shift
null to 0.067, with Pearson r approximately 0.689. White-noise injection was a
null result and per-axis median/IQR calibration overestimated degradation.

In the completed ten-configuration comparison, the synthetic target predicts
real subject-macro target-performance ranking with mean Kendall tau-b 0.677 and
pairwise ordering accuracy 0.839. Robustness-ranking agreement is more modest
(tau-b 0.355; pairwise accuracy 0.678) and heterogeneous across subject folds.
The trivial source-performance ranking baseline reaches tau-b 0.525; synthetic
calibration adds 0.152 on average, but this increment is strongly fold
heterogeneous. The result supports partial predictive validity, not a claim
that scalar gain fully reproduces real device shift. Leaving out the
high-increment fold reduces the gain to tau-b 0.060. See
[`docs/NEURAL_RESULTS.md`](docs/NEURAL_RESULTS.md).

## Research design

```text
HHAR raw audit and checksums
  -> anti-aliased 50 Hz, 5-second windows
  -> subject-disjoint three-fold evaluation
  -> real cross-device degradation matrix
  -> calibrated synthetic device shifts
  -> seven CPU configurations
  -> three raw-sequence neural architectures x three seeds
  -> synthetic-to-real failure and ranking analysis
```

Primary neural architectures:

- Compact 1D CNN;
- DeepConvLSTM;
- ResNet1D.

The final frozen comparison combines these with seven CPU configurations and
reports pooled macro-F1, subject-macro-F1, degradation calibration, Kendall
tau-b, Spearman correlation, top-2 overlap, and pairwise ordering accuracy.

## Reproducibility safeguards

- subject-disjoint train, validation, and test users;
- target test labels excluded from calibration, normalization, and selection;
- protocol commits precede confirmatory execution;
- exact seeds 0, 1, and 2 for neural aggregation;
- fresh train-plus-validation refit after epoch selection;
- atomic checkpoints, SHA-256 verification, and interrupted-run recovery;
- target evaluation blocked until all 135 formal checkpoints pass integrity;
- null and negative results retained.

## Repository layout

- `configs/`: versioned data and split definitions;
- `src/synthshift/`: reusable data, device, training, and evaluation modules;
- `scripts/`: auditable command-line entry points;
- `tests/`: mathematical, integrity, leakage, and lifecycle tests;
- `experiments/`: preregistration, frozen protocols, and job manifests;
- `results/reviewed/`: compact reviewed artifacts only;
- `docs/`: audits, compute notes, limitations, and execution runbooks.

Raw HHAR files, processed arrays, checkpoints, and operational run directories
are intentionally excluded from Git.

## Local setup

Python 3.12 is the recorded development environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,neural]'
pytest
```

The current suite contains 75 tests plus 3 parameterized subtests and does not
require the private local data cache.

## Protocol and execution documents

- [`experiments/PREREGISTRATION.md`](experiments/PREREGISTRATION.md)
- [`experiments/RAW_SEQUENCE_NEURAL_PROTOCOL.md`](experiments/RAW_SEQUENCE_NEURAL_PROTOCOL.md)
- [`docs/CPU_MULTIMODEL_RESULTS.md`](docs/CPU_MULTIMODEL_RESULTS.md)
- [`docs/NEURAL_EXECUTION.md`](docs/NEURAL_EXECUTION.md)
- [`docs/NEURAL_EVALUATION.md`](docs/NEURAL_EVALUATION.md)
- [`docs/NEURAL_RESULTS.md`](docs/NEURAL_RESULTS.md)
- [`docs/FOLD_DIAGNOSTIC.md`](docs/FOLD_DIAGNOSTIC.md)

## Upstream projects

- HHAR dataset: <https://archive.ics.uci.edu/dataset/344/heterogeneity+human+activity+recognition>
- IMU-Bench: <https://github.com/SPICExLAB/IMU-Bench>
- InrtlWrist: <https://huggingface.co/YiquanLi/InrtlWrist>

IMU-Bench/InrtlWrist integration is a planned extension, not part of the
current confirmatory neural grid.

## License

Project-original code is released under the MIT License. HHAR, IMU-Bench,
pretrained weights, and other third-party artifacts retain their own licenses.
