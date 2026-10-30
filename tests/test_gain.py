import unittest

import numpy as np

from synthshift.devices.gain import (
    accelerometer_gain,
    apply_accelerometer_gain,
    gain_ratio,
    still_acceleration_magnitude,
)


class GainTest(unittest.TestCase):
    def test_still_magnitude_estimates_gain_and_ignores_spinning(self) -> None:
        values = np.zeros((2_500, 6))
        values[:, 2] = 1.04 * 9.80665
        values[0, 3] = 2.0
        magnitudes = still_acceleration_magnitude(values)
        self.assertEqual(len(magnitudes), 2_499)
        self.assertAlmostEqual(accelerometer_gain(magnitudes), 1.04, places=6)

    def test_ratio_scales_acceleration_only(self) -> None:
        values = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])
        transformed = apply_accelerometer_gain(values, gain_ratio(1.0, 1.1))
        np.testing.assert_allclose(transformed[0, :3], [1.1, 2.2, 3.3])
        np.testing.assert_allclose(transformed[0, 3:], values[0, 3:])


if __name__ == "__main__":
    unittest.main()
