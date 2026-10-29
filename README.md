# SynthShift

**Can calibrated synthetic IMU shifts predict real cross-device model failures?**

SynthShift is a reproducible research project built around the HHAR dataset and
IMU-Bench model interfaces. It compares a real target phone with a synthetic
proxy produced from a source phone, then asks whether the proxy predicts:

1. cross-device performance degradation;
2. model robustness rankings; and
3. which sensor characteristics explain disagreement.

The primary study uses **phone-to-phone comparisons only**. HHAR watches are not
used for device-only conclusions because their body placement differs from the
phones. No new human-participant data is required for the primary study.

## Status

The repository is at the preregistration and data-audit stage. No experimental
claim has been made yet.

## Research workflow

```text
literature audit -> preregistration -> HHAR audit -> baseline reproduction
-> real device shifts -> calibrated synthetic shifts -> model ranking analysis
```

## Repository layout

- `configs/`: versioned data, model, and experiment definitions.
- `src/synthshift/`: reusable data, device-estimation, perturbation, and evaluation code.
- `scripts/`: command-line entry points.
- `tests/`: mathematical, data-integrity, and leakage tests.
- `experiments/`: preregistration, experiment registry, and negative results.
- `results/`: small derived tables and figures only; raw data and checkpoints are excluded.
- `docs/`: dataset, reproducibility, compute, and limitation notes.
- `third_party/`: pinned upstream references; third-party data/code are not vendored by default.

## Reproducibility principles

- Splits are subject-disjoint unless a preregistered analysis states otherwise.
- Target-device calibration data must be disjoint from final evaluation data.
- Random seeds are repetitions, not additional independent model families.
- Raw HHAR data, model checkpoints, and large run directories are not committed.
- Negative and null results are retained.

## Quick start

Environment setup and executable commands will be added as each stage is
validated. The first planned executable is the HHAR metadata and sampling audit.

## Upstream projects

- HHAR dataset: <https://archive.ics.uci.edu/dataset/344/heterogeneity+human+activity+recognition>
- IMU-Bench: <https://github.com/SPICExLAB/IMU-Bench>
- InrtlWrist: <https://huggingface.co/YiquanLi/InrtlWrist>

## License

Project-original code is released under the MIT License. HHAR, IMU-Bench,
pretrained weights, and other third-party artifacts retain their own licenses.

