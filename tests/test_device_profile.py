import unittest

import numpy as np

from synthshift.devices.profile import affine_parameters, apply_affine


class DeviceProfileTest(unittest.TestCase):
    def test_affine_maps_profile_median_and_iqr(self) -> None:
        source = {"median": [1.0, 2.0], "iqr": [2.0, 4.0]}
        target = {"median": [3.0, -1.0], "iqr": [4.0, 2.0]}
        scale, offset = affine_parameters(source, target)
        values = np.array([[1.0, 2.0], [3.0, 6.0]])
        transformed = apply_affine(values, scale, offset)
        np.testing.assert_allclose(transformed[0], target["median"])
        np.testing.assert_allclose(transformed[1] - transformed[0], [4.0, 2.0])

    def test_rejects_degenerate_profile(self) -> None:
        source = {"median": [0.0], "iqr": [0.0]}
        target = {"median": [0.0], "iqr": [1.0]}
        with self.assertRaisesRegex(ValueError, "IQR"):
            affine_parameters(source, target)
