import unittest

import numpy as np

from synthshift.devices.spectral import (
    GRAVITY_M_S2,
    WINDOW_SAMPLES,
    low_motion_windows,
    spectral_window_features,
)


class SpectralProfileTest(unittest.TestCase):
    def test_windows_never_cross_ineligible_sample(self) -> None:
        values = np.zeros((2 * WINDOW_SAMPLES + 1, 6))
        values[:, 2] = GRAVITY_M_S2
        values[WINDOW_SAMPLES, 3] = 1.0
        windows = low_motion_windows(values)
        self.assertEqual(len(windows), 2)
        self.assertTrue(all(np.max(np.linalg.norm(w[:, 3:], axis=1)) <= 0.15 for w in windows))

    def test_high_frequency_signal_has_more_high_band_power(self) -> None:
        time = np.arange(WINDOW_SAMPLES) / 50.0
        low = np.zeros((WINDOW_SAMPLES, 6))
        high = np.zeros((WINDOW_SAMPLES, 6))
        low[:, :3] = np.sin(2 * np.pi * 3.0 * time)[:, None]
        high[:, :3] = np.sin(2 * np.pi * 14.0 * time)[:, None]
        low_features = spectral_window_features(low, "accelerometer")
        high_features = spectral_window_features(high, "accelerometer")
        self.assertGreater(high_features[3], low_features[3])
        self.assertGreater(high_features[4], low_features[4])


if __name__ == "__main__":
    unittest.main()
