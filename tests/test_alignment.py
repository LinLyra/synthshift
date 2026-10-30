import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from synthshift.data.alignment import direction_stats, load_timestamps, nearest_indices


class AlignmentTest(unittest.TestCase):
    def test_nearest_indices(self) -> None:
        reference = np.array([0, 10, 20])
        query = np.array([2, 9, 18])
        indices, distances = nearest_indices(reference, query)
        np.testing.assert_array_equal(indices, [0, 1, 2])
        np.testing.assert_array_equal(distances, [2, 1, 2])

    def test_load_sorts_and_deduplicates(self) -> None:
        frame = pd.DataFrame(
            {"Creation_Time": [20, 10, 10], "gt": ["walk", "sit", "sit"]}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensor.csv"
            frame.to_csv(path, index=False)
            timestamps, labels, cleaning = load_timestamps(path)
        np.testing.assert_array_equal(timestamps, [10, 20])
        np.testing.assert_array_equal(labels, ["sit", "walk"])
        self.assertEqual(cleaning["reversals"], 1)
        self.assertEqual(cleaning["duplicate_timestamps"], 1)

    def test_direction_stats_reports_tolerance_coverage(self) -> None:
        reference = np.array([0, 10_000_000, 20_000_000])
        query = np.array([1_000_000, 14_000_000, 29_000_000])
        labels = np.array(["walk", "walk", "walk"])
        report = direction_stats(reference, labels, query, labels, (2.5, 5.0))
        self.assertEqual(report["overlap_queries"], 2)
        self.assertEqual(report["matches_by_tolerance_ms"]["2.5"]["coverage"], 0.5)
        self.assertEqual(report["matches_by_tolerance_ms"]["5.0"]["coverage"], 1.0)

    def test_direction_stats_handles_disjoint_ranges(self) -> None:
        labels = np.array(["walk", "walk"])
        report = direction_stats(
            np.array([0, 1]), labels, np.array([10, 11]), labels, (5.0,)
        )
        self.assertEqual(report["overlap_queries"], 0)
        self.assertIsNone(report["nearest_distance_ms_quantiles"])


if __name__ == "__main__":
    unittest.main()
