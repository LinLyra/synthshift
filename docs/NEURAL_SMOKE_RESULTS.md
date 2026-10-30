# Raw-sequence neural smoke and throughput audit

## Decision

The frozen neural training lifecycle works deterministically on the local Apple
Silicon CPU, and memory is not a constraint. The complete confirmatory matrix
should nevertheless run on rented CUDA compute because a serial local CPU run
is estimated to require roughly 13--46 hours depending on selected epochs.
Buying a CPU or GPU is not justified for this experiment.

No target-device score was computed during either benchmark.

## Checks completed

- Source-only fold: `fold_0`, `nexus4_1`.
- Reduced smoke: 384 training and 192 validation windows, stratified across all
  six activities.
- Full-source benchmark: all 2,471 training and 427 validation windows.
- Each architecture completed epoch selection, fresh initialization, exact
  train-plus-validation refit, and checkpoint hashing.
- Repeating the reduced smoke produced identical selected epochs, validation
  scores, and checkpoint SHA-256 values for all three architectures.
- The repository suite passes 63 tests plus 3 parameterized subtests.

## Full-source CPU throughput

The benchmark used one selection epoch and one refit epoch. Times therefore
measure the actual split sizes without turning the benchmark into model
selection.

| Architecture | Parameters | Selection epoch | Refit epoch | Combined |
| --- | ---: | ---: | ---: | ---: |
| Compact CNN | 94,534 | 4.86 s | 5.69 s | 10.55 s |
| DeepConvLSTM | 295,814 | 9.39 s | 9.77 s | 19.16 s |
| ResNet1D | 961,222 | 21.91 s | 22.74 s | 44.65 s |

Peak process resident memory was approximately 0.81 GiB on this macOS runner.
PyTorch 2.15.0 used four CPU threads. PyTorch was compiled with MPS support, but
MPS was unavailable to this executor, so these are CPU timings.

## Runtime envelope

There are 45 runs per architecture: 3 folds x 5 source devices x 3 seeds. A run
performs selection until the best epoch plus up to eight stale epochs, then
refits from scratch for the selected epoch count. Extrapolating the measured
full-source epoch times gives:

- best epochs near 10: approximately 13 local CPU hours in total;
- best epochs near 20: approximately 22 local CPU hours;
- near the 50-epoch cap: approximately 46 local CPU hours.

These are planning estimates, not performance results. CUDA throughput must be
measured with the exact environment before launching all 135 runs. A single
24 GB CUDA GPU is ample for these sub-million-parameter models and batch size
128; capacity is not the reason to rent it, turnaround time is.

## Research interpretation

The low validation macro-F1 values in the two-epoch reduced smoke are not
scientific results and must not be compared between architectures. The smoke
was intentionally too short and too small for convergence. Its admissible
conclusions are limited to data flow, optimization, determinism, memory, and
runtime feasibility.

Artifacts:

- `docs/audits/neural_smoke_cpu.json`
- `docs/audits/neural_full_source_cpu_benchmark.json`
