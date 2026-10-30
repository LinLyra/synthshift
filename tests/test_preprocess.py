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

    def test_only_large_backward_jump_starts_a_new_segment(self) -> None:
        frame = _sensor(
            [2_000_000_000, 1_990_000_000, 500_000_000, 520_000_000],
            ["walk"] * 4,
        )
        segments = split_segments(frame)
        self.assertEqual([len(segment) for segment in segments], [2, 2])
        self.assertTrue(segments[0]["Creation_Time"].is_monotonic_increasing)

    def test_resampling_produces_six_channels(self) -> None:
        acc = _sensor([0, 20_000_000, 40_000_000], ["walk"] * 3)
        gyro = _sensor([0, 10_000_000, 40_000_000], ["walk"] * 3)
        timestamps, values = resample_overlap(acc, gyro, 0, 40_000_000)
        np.testing.assert_array_equal(timestamps, [0, 20_000_000, 40_000_000])
        self.assertEqual(values.shape, (3, 6))

    def test_downsampling_suppresses_above_nyquist_signal(self) -> None:
        source_rate = 200
        timestamps = np.arange(0, 2 * source_rate, dtype=np.int64) * 5_000_000
        signal = np.sin(2 * np.pi * 40 * np.arange(len(timestamps)) / source_rate)
        frame = pd.DataFrame(
            {
                "Creation_Time": timestamps,
                "x": signal,
                "y": signal,
                "z": signal,
                "gt": ["walk"] * len(timestamps),
            }
        )
        _, values = resample_overlap(frame, frame, int(timestamps[0]), int(timestamps[-1]))
        self.assertLess(float(np.sqrt(np.mean(values[:, 0] ** 2))), 0.1)

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
