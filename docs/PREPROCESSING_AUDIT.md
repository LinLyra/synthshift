# HHAR preprocessing audit

## Pipeline

For each user-device stream, the pipeline:

1. preserves acquisition order while detecting boundaries;
2. splits on activity changes, missing labels, forward gaps over 100 ms, and
   backward jumps over one second;
3. stably sorts local timestamp reorder within each remaining segment;
4. collapses duplicate timestamps deterministically;
5. intersects same-activity accelerometer and gyroscope segments;
6. applies a fourth-order zero-phase 20 Hz Butterworth anti-alias filter when
   downsampling, then interpolates each sensor onto a shared 50 Hz integer grid;
7. stores channels as `ax, ay, az, gx, gy, gz` in `float32`.

The first attempt treated every negative timestamp delta as a reset. Inspection
showed that most Nexus 4 reversals were only 5–15 ms and therefore represented
local delivery reorder, while rare resets were many hours. The implementation
and tests were corrected before generating the retained data.

## Usable window counts

Counts below use 5-second windows with 50% overlap. Windowing is computed within
each legal resampled segment, so no window crosses a quality-control boundary.

| Device | Resampled segments | Samples | 5 s windows | 10 s windows | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `nexus4_1` | 960 | 672,447 | 4,231 | 1,786 | primary |
| `nexus4_2` | 935 | 687,892 | 4,415 | 1,900 | primary |
| `s3_1` | 884 | 618,063 | 3,994 | 1,713 | primary |
| `s3_2` | 980 | 685,786 | 4,431 | 1,912 | primary |
| `s3mini_1` | 9,717 | 814,275 | 4,087 | 1,668 | primary, fragmented |
| `s3mini_2` | 914 | 55,932 | 147 | 60 | exploratory only |

The unusually high segment count for `s3mini_1` is caused by frequent gaps over
100 ms. It still contributes a comparable number of legal windows, but this
fragmentation will be reported as a device characteristic and included in
robustness analysis rather than hidden by interpolation across gaps.

Across the five primary devices, each subject contributes approximately
1,900–2,600 five-second windows. Every subject has nonzero coverage for all six
activities. This supports subject-disjoint evaluation without dropping a
participant.

`s3mini_2` has no legal five-second bike or stand windows. It is therefore
excluded from confirmatory model-ranking claims and retained only as a severe
low-coverage timing-shift case study.

## Artifacts

- Canonical local arrays:
  `data/processed/hhar/resampled_50hz_antialias/` (ignored by Git).
- Machine-readable segment manifest:
  `docs/audits/hhar_resampled_antialias_manifest.json`.
- Local array footprint: approximately 134 MiB.
