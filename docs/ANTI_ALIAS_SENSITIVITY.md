# Anti-alias preprocessing sensitivity

The initial 50 Hz corpus used timestamp-aware linear interpolation but no
explicit anti-alias filter. Before synthetic-shift experiments, the pipeline was
corrected to apply a fourth-order zero-phase Butterworth low-pass filter at
20 Hz whenever the estimated native stream rate exceeded 55 Hz.

Both versions were retained and the complete statistical baseline was rerun.

| Summary | Initial interpolation | Anti-aliased canonical |
| --- | ---: | ---: |
| Mean same-device macro-F1 | 0.744 | 0.742 |
| Mean cross-device macro-F1 | 0.686 | 0.686 |
| Mean absolute source-target cell change | — | 0.008 |
| Maximum absolute source-target cell change | — | 0.046 |

The principal finding—substantial, asymmetric real cross-device shift—remains.
The small aggregate change does not justify using the weaker preprocessing
version. All subsequent calibration and synthetic-device experiments use the
anti-aliased corpus.

Canonical matrix:

| Source \ Target | nexus4_1 | nexus4_2 | s3_1 | s3_2 | s3mini_1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nexus4_1` | 0.777 | 0.731 | 0.708 | 0.592 | 0.593 |
| `nexus4_2` | 0.773 | 0.793 | 0.772 | 0.713 | 0.686 |
| `s3_1` | 0.712 | 0.726 | 0.746 | 0.677 | 0.647 |
| `s3_2` | 0.615 | 0.768 | 0.744 | 0.776 | 0.684 |
| `s3mini_1` | 0.532 | 0.666 | 0.653 | 0.724 | 0.618 |

Machine-readable canonical result:
`results/reviewed/statistical_baseline_antialias.json`.
