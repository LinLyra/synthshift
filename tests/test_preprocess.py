import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from synthshift.data.preprocess import (
    preprocess_pair,
    resample_overlap,
    split_segments,
    window_count,
)


def _sensor(timestamps: list[int], labels: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Creation_Time": timestamps,
            "x": np.arange(len(timestamps), dtype=float),
            "y": np.arange(len(timestamps), dtype=float),
            "z": np.arange(len(timestamps), dtype=float),
            "gt": labels,
        }
    )


class PreprocessTest(unittest.TestCase):
    def test_segmentation_precedes_sorting_and_drops_missing_labels(self) -> None:
        frame = _sensor(
            [0, 20_000_000, 40_000_000, 10_000_000, 30_000_000, 50_000_000],
            ["walk", "walk", None, "sit", "sit", "sit"],
        )
        segments = split_segments(frame)
        self.assertEqual([segment["gt"].iloc[0] for segment in segments], ["walk", "sit"])
        self.assertEqual([len(segment) for segment in segments], [2, 3])

    def test_resampling_produces_six_channels(self) -> None:
        acc = _sensor([0, 20_000_000, 40_000_000], ["walk"] * 3)
        gyro = _sensor([0, 10_000_000, 40_000_000], ["walk"] * 3)
        timestamps, values = resample_overlap(acc, gyro, 0, 40_000_000)
        np.testing.assert_array_equal(timestamps, [0, 20_000_000, 40_000_000])
        self.assertEqual(values.shape, (3, 6))

    def test_window_count(self) -> None:
        self.assertEqual(window_count(127, 128, 64), 0)
        self.assertEqual(window_count(128, 128, 64), 1)
        self.assertEqual(window_count(192, 128, 64), 2)

    def test_preprocess_pair_writes_reproducible_record(self) -> None:
        frame = _sensor([0, 20_000_000, 40_000_000], ["walk"] * 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            acc = root / "a__phone.csv"
            gyro = root / "gyro.csv"
            frame.to_csv(acc, index=False)
            frame.to_csv(gyro, index=False)
            records = preprocess_pair(acc, gyro, root / "output")
            bundle = np.load(root / "output" / records[0]["file"])
            self.assertEqual(bundle["values"].shape, (3, 6))
            self.assertEqual(records[0]["label"], "walk")


if __name__ == "__main__":
    unittest.main()
