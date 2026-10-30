import unittest

import numpy as np

from synthshift.devices.clipping import clip_accelerometer, clipping_diagnostics


class ClippingTest(unittest.TestCase):
    def test_clips_acceleration_and_preserves_gyroscope(self) -> None:
        values = np.array([[30.0, -30.0, 1.0, 0.1, 0.2, 0.3]])
        transformed = clip_accelerometer(values, rail_m_s2=20.0)
        np.testing.assert_allclose(transformed[0, :3], [20.0, -20.0, 1.0])
        np.testing.assert_allclose(transformed[0, 3:], values[0, 3:])

    def test_diagnostics_use_scalar_and_window_denominators(self) -> None:
        windows = np.zeros((2, 4, 6))
        windows[0, 0, 0] = 30.0
        result = clipping_diagnostics(windows, rail_m_s2=20.0)
        self.assertAlmostEqual(result["changed_scalar_fraction"], 1.0 / 24.0)
        self.assertAlmostEqual(result["changed_window_fraction"], 0.5)
        self.assertEqual(result["maximum_absolute_after"], 20.0)


if __name__ == "__main__":
    unittest.main()
