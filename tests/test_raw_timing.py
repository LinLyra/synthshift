import unittest

import numpy as np

from synthshift.data.raw_timing import timing_metrics_from_timestamps
from synthshift.eval.raw_timing_audit import _equal_user_profile


class RawTimingTest(unittest.TestCase):
    def test_acquisition_order_metrics_retain_gap_duplicate_and_reorder(self) -> None:
        timestamps = np.array(
            [0, 20_000_000, 40_000_000, 40_000_000, 30_000_000, 230_000_000],
            dtype=np.int64,
        )
        result = timing_metrics_from_timestamps(timestamps)
        self.assertEqual(result["sample_count"], 6)
        self.assertAlmostEqual(result["median_positive_interval_ms"], 20.0)
        self.assertAlmostEqual(result["duplicate_fraction"], 0.2)
        self.assertAlmostEqual(result["reorder_fraction"], 0.2)
        self.assertAlmostEqual(result["gap_fraction"], 0.2)

    def test_equal_user_profile_does_not_sample_weight(self) -> None:
        names = (
            "sample_count", "observed_duration_s", "median_positive_interval_ms",
            "effective_median_rate_hz", "normalized_interval_mad",
            "positive_interval_p95_ms", "positive_interval_p99_ms",
            "duplicate_fraction", "reorder_fraction", "severe_backward_fraction",
            "gap_fraction", "gaps_per_observed_minute", "burst_fraction",
        )
        first = {name: 0.0 for name in names}
        second = {name: 2.0 for name in names}
        result = _equal_user_profile({"a": first, "h": second})
        self.assertTrue(all(value == 1.0 for value in result.values()))


if __name__ == "__main__":
    unittest.main()
