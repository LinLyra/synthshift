import unittest

import numpy as np

from synthshift.data.features import extract_features


class FeatureTest(unittest.TestCase):
    def test_feature_shape_and_finiteness(self) -> None:
        time = np.linspace(0, 1, 250)
        window = np.column_stack([time + axis for axis in range(6)])
        features = extract_features(window)
        self.assertEqual(features.shape, (78,))
        self.assertTrue(np.isfinite(features).all())

    def test_rejects_wrong_channel_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "6"):
            extract_features(np.zeros((250, 3)))
