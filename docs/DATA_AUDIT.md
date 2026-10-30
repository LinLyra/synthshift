# HHAR data audit

**Audit status:** initial raw-file audit; accelerometer/gyroscope alignment and
confirmatory split construction are not yet complete.

## Provenance and storage

- Official UCI archive: 822,098,071 bytes (784 MiB).
- SHA-256: `8c9f8dd5f50074feaad2c5cec5d927a4fc8366324de26079629a3de3aef0e124`.
- Extracted local footprint: approximately 4.0 GiB, including the retained
  nested official archives.
- Raw data and derived training artifacts are excluded from Git. The compact
  source manifest and audit reports are versioned.

## Activity-file coverage

| Sensor | Rows | Users | Device instances | Device models | Missing activity label |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phone accelerometer | 13,062,475 | 9 | 8 | 4 | 1,783,200 (13.65%) |
| Phone gyroscope | 13,932,632 | 9 | 6 | 3 | 1,869,627 (13.42%) |

The accelerometer file contains two instances each of Nexus 4, Galaxy S3,
Galaxy S3 Mini, and Galaxy S+. The gyroscope file does not contain either
`samsungold` (Galaxy S+) instance. Consequently, an accelerometer-only analysis
can cover eight devices, but a six-axis analysis can cover at most six.

`s3mini_2` has only 95,703 accelerometer rows and 95,650 gyroscope rows, compared
with roughly 1.4–3.2 million rows for each other sensor/device combination. All
six named activities are present, but this device must not be treated as a
full-coverage peer without a duration and overlap analysis.

The accelerometer stream for user `f` on `s3_1` is absent while the gyroscope
stream is present. That user-device stream cannot contribute six-axis windows.

There are 53 user-device streams with both sensors: nine for each six-axis
device except `s3_1`, which has eight. Before exact timestamp matching, the
conservative labelled-duration upper bounds are approximately 3.5–4.2 hours per
well-covered device, versus only 20.7 minutes total for `s3mini_2`. This confirms
that `s3mini_2` should not drive the confirmatory device-ranking result.

## Raw timing diagnostics

Creation timestamps are consistent with nanoseconds. Median effective rates
estimated independently within each user-device stream are approximately:

| Device | Accelerometer | Gyroscope |
| --- | ---: | ---: |
| `nexus4_1`, `nexus4_2` | 198.6 Hz | 198.6 Hz |
| `s3_1` | 98.9 Hz | 200.0 Hz |
| `s3_2` | 98.8 Hz | 200.0 Hz |
| `s3mini_1` | 99.9 Hz | 99.9 Hz |
| `s3mini_2` | median 71.6 Hz | median 71.8 Hz |
| `samsungold_1`, `samsungold_2` | 50.0 Hz | unavailable |

These are diagnostic estimates, not nominal-rate labels. They are based on up
to 20,000 consecutive intervals per user-device stream. Duplicate or reversed
timestamps exist, especially for the Nexus 4 streams, and must be handled by an
explicit cleaning rule rather than silently sorted away.

## Consequences for the study

1. The confirmatory six-axis device set is not yet fixed. The five devices with
   substantial two-sensor coverage (`nexus4_1`, `nexus4_2`, `s3_1`, `s3_2`, and
   `s3mini_1`) are candidates; `s3mini_2` remains exploratory until aligned
   duration and activity coverage are quantified.
2. Galaxy S+ remains useful for an accelerometer-only replication but cannot be
   included in a six-axis InrtlWrist comparison.
3. Sensor-rate heterogeneity is real and large. Resampling is therefore both a
   preprocessing decision and a candidate explanatory corruption; the same
   transformation must not be counted twice.
4. Rows with missing activity labels are excluded from supervised scoring but
   may be eligible for explicitly declared unlabelled calibration. They must
   never leak into final evaluation-derived calibration statistics.
5. Device inclusion and train/calibration/test partitions will be frozen only
   after timestamp overlap, usable duration, and class coverage are measured.

## Remaining checks before preregistration freeze

- Match accelerometer and gyroscope streams by user, device, and timestamp.
- Quantify usable labelled duration and contiguous segments by activity.
- Determine whether timestamp reversals are duplicate delivery, clock reset, or
  ordering artifacts.
- Verify target calibration can be separated temporally from final evaluation.
- Construct and hash the final subject/device split manifests.

Machine-readable reports:

- `docs/audits/hhar_phone_accelerometer.json`
- `docs/audits/hhar_phone_gyroscope.json`
- `docs/audits/hhar_phone_coverage.json`

The subsequent timestamp-matching results and preprocessing decisions are in
`docs/ALIGNMENT_AUDIT.md`.
