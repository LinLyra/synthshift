import unittest

import numpy as np

from synthshift.data.sync import estimate_pair, prepare_stream


def _motion(delay_ms: int = 0, *, duration_s: float = 50.0, gap: bool = False, seed: int = 0):
    generator = np.random.default_rng(seed)
    t = np.arange(int(duration_s * 50)) / 50.0
    if gap:
        t = t[(t < 10.0) | (t > 12.0)]
    kernel = np.exp(-0.5 * (np.arange(-30, 31) / 5.0) ** 2)
    kernel /= kernel.sum()
    magnitude = 9.81 + np.convolve(generator.normal(size=t.size), kernel, mode="same")
    time_ms = np.rint(1_424_696_633_908 + t * 1000).astype(np.int64)
    return time_ms, magnitude, time_ms + np.int64(delay_ms), magnitude.copy()


class SyncTest(unittest.TestCase):
    def test_recovers_delay_sign_and_half_session_agreement(self) -> None:
        reference_time, reference_magnitude, other_time, other_magnitude = _motion(235)
        estimate = estimate_pair(reference_time, reference_magnitude, other_time, other_magnitude)
        self.assertTrue(estimate["usable"])
        self.assertLessEqual(abs(estimate["offset_ms"] + 235), 5)
        self.assertLessEqual(estimate["half_offset_abs_diff_ms"], 10)
        self.assertGreater(estimate["peak_correlation"], estimate["correlation_at_zero"])

    def test_gap_does_not_move_the_recovered_offset(self) -> None:
        reference_time, reference_magnitude, other_time, other_magnitude = _motion(180, gap=True)
        estimate = estimate_pair(reference_time, reference_magnitude, other_time, other_magnitude)
        self.assertLessEqual(abs(estimate["offset_ms"] + 180), 5)

    def test_independent_noise_is_not_usable(self) -> None:
        generator = np.random.default_rng(1)
        time_ms = 1_424_696_633_908 + np.arange(2500) * 20
        estimate = estimate_pair(
            time_ms,
            generator.normal(size=time_ms.size) + 9.81,
            time_ms + 200,
            generator.normal(size=time_ms.size) + 9.81,
        )
        self.assertFalse(estimate["usable"])

    def test_disjoint_clocks_have_no_offset(self) -> None:
        time_ms = 1_424_696_633_908 + np.arange(2500) * 20
        magnitude = np.sin(time_ms / 1000.0)
        estimate = estimate_pair(time_ms, magnitude, time_ms + 10_000_000, magnitude)
        self.assertIsNone(estimate["offset_ms"])
        self.assertFalse(estimate["usable"])

    def test_prepare_stream_sorts_without_reading_a_label_column(self) -> None:
        time_ms, magnitude, cleaning = prepare_stream(
            np.array([30, 10, 10, 20]),
            np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 4.0]]),
        )
        np.testing.assert_array_equal(time_ms, [10, 20, 30])
        self.assertEqual(cleaning["duplicate_timestamps"], 1)
        self.assertEqual(cleaning["reversals"], 1)
        self.assertGreater(magnitude[0], 0)


if __name__ == "__main__":
    unittest.main()
