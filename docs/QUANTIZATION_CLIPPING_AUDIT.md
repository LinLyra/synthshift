# Raw quantization and clipping-signature audit

**Status:** exploratory audit. Protocol commit `22bf262`; implementation commit
`3b0854d`. The reported spacings describe exported CSV values and are not
claimed as native ADC resolution.

## Stable representation signatures

The empirical accelerometer unique-value spacing is highly reproducible across
calibration users and folds:

| Device | Median lower-tail unique spacing |
| --- | ---: |
| `nexus4_1`, `nexus4_2` | about 0.0011900 m/s² |
| `s3_1`, `s3_2` | about 0.0095766 m/s² |
| `s3mini_1` | about 0.0011969 m/s² |

Despite this clear device fingerprint, spacing differences have little stable
relationship with gain residuals. The target-minus-source acceleration spacing
has Pearson -0.235 and Spearman 0.067 with signed residual. Quantization spacing
therefore does not currently justify a score-producing perturbation.

## `s3mini_1` clipping signature

`s3mini_1` repeatedly reaches an acceleration magnitude near exactly ±19.613
m/s² on individual axes, consistent with an approximately ±2g exported rail.
Across folds its maximum extreme occupancy is roughly 0.00004–0.00046 and its
maximum exact-extreme run is 3–5 samples. Other primary devices generally have
extreme occupancy around 0.000003–0.000011 and runs of 1–2 samples, with extrema
that vary substantially by participant rather than repeating at a fixed rail.

Directional clipping summaries correlate with signed gain residual:

| Target minus source factor | Pearson | Spearman |
| --- | ---: | ---: |
| Exact extreme occupancy | 0.595 | 0.547 |
| Within-one-spacing extreme occupancy | 0.598 | 0.549 |
| Longest exact extreme run | 0.671 | 0.525 |

The longest-run Pearson values by fold are 0.843, 0.531, and 0.623. However,
after removing every transfer involving `s3mini_1`, the same correlation becomes
-0.065 (Spearman -0.088). Extreme occupancy similarly falls from 0.595 to
0.051.

## Decision

Quantization is retained as a reproducible device fingerprint but not a residual
explanation. Clipping is a credible `s3mini_1`-specific mechanism and may be
tested as a predeclared severe-device case study.

The transformation is inherently one-sided: a higher-range source can be
clipped toward an `s3mini_1` target, but already clipped `s3mini_1` samples
cannot be reconstructed into an unclipped target. It must not be scored as if
the reverse transformation were available.

Machine-readable artifact:
`results/reviewed/quantization_clipping_audit.json`.
