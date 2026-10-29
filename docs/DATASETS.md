# Datasets

## HHAR

Primary source: <https://archive.ics.uci.edu/dataset/344/heterogeneity+human+activity+recognition>

The primary study uses phone accelerometer and gyroscope recordings. Raw data are
not redistributed by this repository. Download them from UCI and retain the
original archive and checksum outside Git.

Expected activity files include:

- `Phones_accelerometer.csv`
- `Phones_gyroscope.csv`
- `Watch_accelerometer.csv`
- `Watch_gyroscope.csv`

The phone data are primary. Watch data are excluded from device-only conclusions
because phones and watches have different body placements.

Before training, the audit must verify:

- required columns and label values;
- user, model, and physical-device counts;
- timestamp units, monotonicity, gaps, and effective rates;
- accelerometer/gyroscope matching coverage;
- activity duration by user/device;
- missing and null labels;
- whether nominally paired device streams can be aligned credibly.

## Data directory

Set the path without committing it:

```bash
export SYNTHSHIFT_DATA_ROOT=/absolute/path/to/synthshift-data
```

