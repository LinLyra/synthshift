# Raw-sequence neural model protocol

**Status:** frozen Phase-B protocol. Commit before installing the training
dependency, implementing models, or running any neural target evaluation.

## Purpose

Test whether scalar acceleration gain predicts real cross-device degradation
and model ranking for architectures that learn directly from 5-second six-axis
sequences rather than the shared 78-dimensional handcrafted representation.

## Input and separation

- Input shape: 250 samples by 6 channels at 50 Hz.
- Use the same legal windows, subject folds, five primary phone instances, and
  labels as the CPU audit.
- Train one source-device model per fold and architecture.
- Gain profiles use both named training calibration users without labels.
- Test users and real target scores never enter normalization, epoch selection,
  gain estimation, or training.
- No random crop, rotation, jitter, masking, or other augmentation is used in
  the primary experiment; gain remains the only synthetic factor.

## Fixed normalization

For epoch selection, estimate one mean and standard deviation per channel from
the source-device training users only. For the final refit, re-estimate them
from source train plus validation users. Clamp standard deviations below 1e-6
to 1e-6. Apply gain in physical units before source normalization:

```text
acc_fake = acc_source * (gain_target / gain_source)
normalized_fake = (fake - source_mean) / source_std
```

The same source normalization is applied to untouched source and real target
windows.

## Fixed architectures

### Compact 1D CNN

- Conv1d 6→64, kernel 7, same padding; batch norm; ReLU;
- Conv1d 64→128, kernel 5; batch norm; ReLU; max-pool 2;
- Conv1d 128→128, kernel 3; batch norm; ReLU;
- global average pooling and linear classifier.

### DeepConvLSTM

- four Conv1d layers, 64 channels, kernel 5, same padding, ReLU;
- transpose to time-major features;
- two-layer unidirectional LSTM with hidden size 128;
- last time-step representation and linear classifier.

### ResNet1D

- stem Conv1d 6→64, kernel 7, batch norm, ReLU;
- three residual stages with channels 64, 128, and 256;
- two basic blocks per stage, kernel 3; first block of stages 2 and 3 uses
  stride 2 with a projection shortcut;
- global average pooling and linear classifier.

No architecture width, depth, kernel, dropout, or activation search is allowed
after target evaluation.

## Fixed optimization

- seeds: 0, 1, 2;
- loss: class-weighted cross entropy, inverse source-training frequency with
  weights normalized to mean one;
- optimizer: AdamW, learning rate 1e-3, weight decay 1e-4;
- batch size: 128;
- maximum selection epochs: 50;
- early stopping: validation macro-F1, patience 8, minimum improvement 1e-4;
- gradient norm clipping: 1.0;
- no learning-rate search or scheduler;
- retain the earliest epoch on an exact validation tie.

After selecting a best epoch for a fold/source/architecture/seed, initialize a
fresh model with the same seed and train for exactly that many epochs on train
plus validation users. This final refit is the only model evaluated on targets.

## Reproducibility and failure handling

Seed Python, NumPy, and PyTorch. Request deterministic algorithms where the
selected backend supports every used operation. Store package versions, backend,
device name, parameter count, selected epoch, training time, and checkpoint
hash. A backend incompatibility is recorded; it does not authorize changing the
architecture. Smoke tests may use a reduced subset but may not inspect target
scores.

## Evaluation

For every final model, evaluate:

- untouched source test windows;
- each real target device;
- each gain-transformed synthetic target using the source windows.

Report pooled macro-F1 and subject-macro-F1 exactly as in the CPU audit. Aggregate
the three seeds within architecture before degradation prediction and ranking.

## Outcomes

For each architecture, report 60 off-diagonal real/synthetic degradation cells:
MAE, RMSE, Pearson, Spearman, calibration slope/intercept, R-squared, and sign
agreement.

Final combined ranking uses the seven committed CPU configurations plus the
three seed-aggregated neural architectures:

- primary: lower real degradation versus lower synthetic degradation;
- secondary: higher real target score versus higher synthetic target score;
- metrics: Kendall tau-b, Spearman, top-2 overlap, and pairwise ordering
  accuracy, for pooled and subject-macro scoring.

The CPU artifact is immutable input to the combined ranking; CPU models are not
retrained after neural results are seen.

## Compute stages

1. Build and checksum a local float32 raw-window cache.
2. Install a pinned PyTorch dependency and run unit tests.
3. Run CPU or Apple-MPS smoke tests on one fold/source with a small training
   subset; do not compute target scores.
4. Estimate full runtime and memory.
5. Run the complete 135 selection/refit jobs on a rented CUDA GPU if local MPS
   throughput is not practical.

No CPU or GPU purchase is required. Cloud rental begins only after stages 1–4
pass and the exact environment is recorded.
