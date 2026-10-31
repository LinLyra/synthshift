# Neural execution and recovery

The formal source-training grid is frozen in
`experiments/neural_job_manifest.json`. It contains exactly 135 unique jobs:
three folds, five source devices, three architectures, and three seeds.

## Safety properties

- Formal jobs reject epoch, patience, and row-limit overrides.
- Training and validation use only the named source device and the frozen
  subject split.
- Target-device windows are not loaded by the training executor.
- Checkpoints and metadata are written atomically.
- `metadata.json` is the completion marker and records the checkpoint SHA-256.
- Resume verifies that checksum before skipping a completed job.
- A crash before metadata completion causes that one job to restart from the
  beginning; completed jobs are retained.
- Smoke runs use a separate `runs/neural/smoke/` namespace and therefore cannot
  be mistaken for formal runs.

## CUDA preflight

On a rented host, install the pinned project environment, copy the checksummed
feature and raw-window files, and first run exactly one formal job:

```bash
python scripts/run_neural_training_queue.py \
  experiments/neural_job_manifest.json \
  data/processed/hhar/features_5s_antialias.npz \
  data/processed/hhar/raw_windows_5s_antialias.npy \
  configs/splits/hhar_subject_folds.json \
  runs/neural --device cuda --limit 1
```

Inspect its environment, runtime, selected epoch, learning history, checkpoint
hash, and source validation behavior before removing `--limit 1`. This
preflight is part of execution QA, not an opportunity to tune the protocol.

## Full queue

After preflight approval, rerun the same command without `--limit`. The first
completed job will be checksum-verified and skipped. Filters such as
`--architecture resnet1d` or `--fold fold_0` may divide operational batches,
but do not change any training setting.

Formal outputs remain ignored by Git because checkpoints are large. Compact
reviewed result tables and the final audit will be exported separately after
all expected jobs pass integrity checks.
