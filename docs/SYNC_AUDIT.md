# HHAR cross-phone Arrival_Time audit

**Status:** diagnostic completed. A constant clock offset is not stable enough
to estimate a 3×3 sensor map. No transform was fit, and test users were not
read.

Within-phone accelerometer–gyroscope alignment stays on `Creation_Time`. That
clock is local to each phone. This audit uses `Arrival_Time` only to ask
whether two phones in the same pouch share a usable common timeline.

## Method

The official phone accelerometer file was streamed for the six calibration
users named in `configs/splits/hhar_subject_folds.json` and the five primary
devices. Activity labels were not loaded. Each stream was sorted on
`Arrival_Time`, duplicate milliseconds were collapsed, and the signal was the
acceleration magnitude.

Magnitude was averaged in 50 ms bins, high-pass filtered with a 2 s window, and
restricted to bins at or above the median absolute value. The searched offset
is the constant added to the second device's `Arrival_Time`. The coarse peak
uses FFT correlation inside ±120 s. A direct search then moves that peak in
5 ms steps over ±200 ms. The recording was also split at the midpoint of the
raw overlap, and the offset was estimated again on each half.

A user estimate is usable only if all of the following hold:

- raw overlap at least 20 s and dynamic overlap at least 5 s;
- peak correlation at least 0.30;
- peak margin over the next peak at least 1 s away at least 0.10;
- the two half-session offsets differ by at most 50 ms;
- the peak is inside the search window.

A fold-pair is stable only if both calibration users are usable and their
offsets differ by at most 50 ms. These thresholds were coded before the HHAR
run.

Nearest-neighbour spacing is reported as sampling density. The same spacing
appears at offset zero, so a neighbour within 20 ms is not evidence that the
offset recovered the same motion.

## Clock check

After cleaning, every calibration stream spans 31–39 minutes, and phones worn
by the same user agree on that span to within a few seconds. `Arrival_Time`
has no acquisition-order reversals in these streams. Nexus 4 arrival times have
a 5 ms median step and roughly 40,000–48,000 duplicate milliseconds per user;
Samsung phones have a 10 ms median step and a few hundred duplicates. Duplicate
milliseconds were collapsed before correlation.

## Result

Of 60 user-pair estimates, 19 are usable. Of 30 fold-pairs, 3 are stable, all
in fold 1 (users `a` and `h`):

| Pair | Level | Offset, user a | Offset, user h | Cross-user difference | Peak correlations |
| --- | --- | ---: | ---: | ---: | --- |
| `nexus4_1` → `s3_1` | model | -35 ms | -80 ms | 45 ms | 0.644, 0.532 |
| `nexus4_1` → `s3_2` | model | 155 ms | 115 ms | 40 ms | 0.770, 0.796 |
| `s3_1` → `s3_2` | instance | 195 ms | 200 ms | 5 ms | 0.838, 0.798 |

`s3mini_1` has no stable pair. The same three pairs are not stable for the
calibration users in fold 0 or fold 2.

Twelve fold-pairs have cross-user offset differences of at most 50 ms, but
nine of those still fail because a half-session peak moves. Across all 60
half-session differences, the median is 310 ms, 19 are at most 50 ms, and 5
exceed 10 s. Large half-session gaps are usually one half locking onto a
different peak while the other half stays near the full-session offset. That
is enough to reject one constant lag for the whole recording.

The Nexus instance pair illustrates the pattern. Full-session offsets are
-345, -350, -310, -305, -340, and -320 ms for users `b`, `g`, `a`, `h`, `c`,
and `i`. Correlation at that lag is 0.30–0.81, while correlation at lag zero
is negative or near zero. Only users `b` and `i` also keep both halves inside
the 50 ms band.

After the chosen offset, the median nearest-neighbour gap is 2 ms on Nexus
pairs and 3 ms otherwise, and 82–92% of reference samples have a neighbour
within 20 ms. The lag-zero figures are the same. Dense overlap and correct
motion alignment are separate facts.

## Decision

Paired-sample 3×3 maps are not estimated from this audit. A map fit on the
three fold-1 pairs would be a subset chosen after seeing the gate. Later
paired calibration, if it is attempted, needs a new pre-specified rule for
piecewise offsets or for segments where both halves and both calibration users
agree. Until then, device comparisons stay with the marginal affine result and
with factors that do not require sample-level cross-phone pairing.

Machine-readable report: `docs/audits/hhar_phone_sync.json`.
