import unittest

import numpy as np

from synthshift.devices.gyro_bias import (
    apply_gyroscope_bias,
    gyroscope_bias,
    low_motion_gyroscope,
)


class GyroscopeBiasTest(unittest.TestCase):
    def test_gate_and_componentwise_median(self) -> None:
        values = np.zeros((2_501, 6))
        values[:, 2] = 9.80665
        values[:, 3:] = [0.01, -0.02, 0.03]
        values[0, 0] = 20.0
        values[1, 3] = 1.0
        samples = low_motion_gyroscope(values)
        self.assertEqual(len(samples), 2_499)
        np.testing.assert_allclose(gyroscope_bias(samples), [0.01, -0.02, 0.03])

    def test_transform_changes_gyroscope_only(self) -> None:
        values = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])
        transformed = apply_gyroscope_bias(values, np.array([0.01, -0.02, 0.03]))
        np.testing.assert_allclose(transformed[0, :3], values[0, :3])
        np.testing.assert_allclose(transformed[0, 3:], [0.11, 0.18, 0.33])

    def test_rejects_insufficient_samples(self) -> None:
        with self.assertRaises(ValueError):
            gyroscope_bias(np.zeros((1_999, 3)))


if __name__ == "__main__":
    unittest.main()
