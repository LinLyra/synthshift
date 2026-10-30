# HHAR phone sensor alignment audit

**Status:** feasibility audit completed; this report selects preprocessing
defaults but is not itself a training result.

## Method

The raw accelerometer and gyroscope files were sharded in one pass by user and
physical device. Within each stream, `Creation_Time` was parsed as nanoseconds,
stably sorted, and duplicate timestamps were deterministically collapsed. For
each accelerometer timestamp, the nearest gyroscope timestamp was located, and
the reverse direction was measured separately. Coverage was evaluated at 2.5,
5, 10, 20, and 40 ms without interpolating sensor values.

Sorting is necessary because the released CSV order contains reversals,
especially where recording segments were concatenated. It does not authorize
bridging recording gaps: final window construction must split at clock
reversals in acquisition order, activity transitions, missing labels, and long
time gaps before resampling.

## Results

Mean accelerometer-to-gyroscope coverage across user streams:

| Device | User streams | 5 ms | 10 ms | Mean nearest-distance p95 |
| --- | ---: | ---: | ---: | ---: |
| `nexus4_1` | 9 | 99.994% | 99.998% | 1.02 ms |
| `nexus4_2` | 9 | 99.993% | 99.998% | 1.01 ms |
| `s3_1` | 8 | 99.988% | 99.990% | 2.38 ms |
| `s3_2` | 9 | 100.000% | 100.000% | 2.39 ms |
| `s3mini_1` | 9 | 99.053% | 99.938% | 3.63 ms |
| `s3mini_2` | 9 | 23.646% | 98.643% | 7.12 ms |

At 10 ms, mean activity-label agreement among labelled matches exceeds 99.99%
for the five well-covered devices. Reverse gyroscope-to-accelerometer coverage
is also at least 99.5% for the main devices; the slight reduction for Galaxy S3
is expected because its gyroscope is approximately 200 Hz while its
accelerometer is approximately 99 Hz.

No duplicate accelerometer timestamps were observed. The Galaxy S3 gyroscope
streams contained a small number of duplicate timestamps (490 for `s3_1`, 429
for `s3_2` across users), which will be collapsed before resampling.

## Decisions supported by this audit

1. The confirmatory six-axis device set is `nexus4_1`, `nexus4_2`, `s3_1`,
   `s3_2`, and `s3mini_1`, subject to final window-count checks.
2. `s3mini_2` is exploratory only because its total duration is small and its
   native timing differs materially from its sibling instance.
3. The canonical model input rate will be 50 Hz. This does not invent a claim
   that devices natively sampled at 50 Hz; it provides a common rate at or below
   the well-covered devices' observed rates.
4. A 10 ms nearest-neighbour threshold is retained as a pairing diagnostic, not
   as the final fusion algorithm. Training inputs will be created by segmenting
   first and resampling each sensor onto a common 50 Hz grid.
5. Windows that cross a recording gap, clock reversal, activity transition, or
   missing-label region will be rejected.

## Remaining validation

- Quantify post-resampling windows per user, device, and activity.
- Confirm interpolation never crosses the segment boundaries above.
- Measure the fraction of windows removed by each quality-control rule.
- Freeze subject folds, target calibration partitions, and device-pair rules.

Machine-readable report: `docs/audits/hhar_phone_alignment.json`.
