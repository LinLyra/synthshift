# One-sided `s3mini_1` clipping case study

**Status:** frozen-protocol exploratory null result. Protocol commit `417f81f`;
implementation commit `345a461`. The ±2g rail was not tuned after scoring.

## Design

The 12 evaluated cells contain all three folds and the four non-`s3mini_1`
source devices with `s3mini_1` as target. Scalar acceleration gain is applied
first, followed by clipping at ±19.6133 m/s². Reverse transfers are not
evaluated because clipping cannot reconstruct lost amplitude.

## Result

| Forecast | Degradation MAE | RMSE | Pearson r | Mean predicted degradation |
| --- | ---: | ---: | ---: | ---: |
| Gain only | 0.07903 | 0.11466 | 0.6250 | 0.04685 |
| Gain + fixed clipping | 0.07936 | 0.11485 | 0.6253 | 0.04652 |

Clipping increases mean absolute error by 0.00034. Zero of 12 cells improve,
nine are score-identical, and three worsen.

Only 0.0186% of scalar acceleration values are changed on average. About 2.58%
of windows contain at least one changed value, but most changes do not alter the
statistical model's prediction. Galaxy S3 sources never exceed the rail after
gain. Nexus 4 sources exceed it sparsely; the score changes move predicted
degradation slightly farther from the real value.

## Interpretation

The repeated ±19.613 m/s² rail is a credible raw representation signature of
`s3mini_1`, but it is too rarely activated to explain this model's target-domain
degradation. Extreme occupancy correlated with residuals because it identified
the `s3mini_1` domain, not because hard clipping was the causal factor.

The rail will not be lowered or made axis-specific after observing this result.
Clipping is closed as an explanatory factor for the current linear model, while
remaining a possible stress test for high-dynamic-range motion tasks.

Machine-readable artifact:
`results/reviewed/s3mini_clipping_case_study.json`.
