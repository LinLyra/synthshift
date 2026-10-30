import tempfile
import unittest
from pathlib import Path

import pandas as pd

from synthshift.data.coverage import pair_profiles, profile_csv


class CoverageProfileTest(unittest.TestCase):
    def test_does_not_bridge_large_or_reversed_gaps(self) -> None:
        frame = pd.DataFrame(
            {
                "Creation_Time": [0, 100_000_000, 2_000_000_000, 1_900_000_000],
                "User": ["a"] * 4,
                "Device": ["phone_1"] * 4,
                "gt": ["walk"] * 4,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensor.csv"
            frame.to_csv(path, index=False)
            stream = profile_csv(path, chunksize=2)["streams"]["a|phone_1"]

        self.assertEqual(stream["duration_seconds_by_label"]["walk"], 0.1)
        self.assertEqual(stream["large_gaps"], 1)
        self.assertEqual(stream["nonpositive_intervals"], 1)

    def test_pairing_reports_unmatched_streams(self) -> None:
        acc = {"streams": {"a|one": {"duration_seconds_by_label": {"walk": 5.0}}}}
        gyro = {
            "streams": {
                "a|one": {"duration_seconds_by_label": {"walk": 4.0}},
                "b|two": {"duration_seconds_by_label": {"walk": 3.0}},
            }
        }
        paired = pair_profiles(acc, gyro)
        self.assertEqual(paired["paired_stream_count"], 1)
        self.assertEqual(paired["gyroscope_only_streams"], ["b|two"])
        self.assertEqual(
            paired["paired_streams"]["a|one"]["duration_upper_bound_seconds_by_label"]["walk"],
            4.0,
        )


if __name__ == "__main__":
    unittest.main()
