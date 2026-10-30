import unittest

import numpy as np

from synthshift.data.quantization import _longest_true_run, channel_representation_metrics


class QuantizationProfileTest(unittest.TestCase):
    def test_fixed_step_and_extreme_run(self) -> None:
        values = np.array([-2.0, -2.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.0])
        result = channel_representation_metrics(values)
        self.assertAlmostEqual(result["unique_spacing_p10"], 1.0)
        self.assertAlmostEqual(result["unique_spacing_p50"], 1.0)
        self.assertEqual(result["longest_exact_extreme_run"], 3)
        self.assertAlmostEqual(result["extreme_symmetry_error"], 0.0)

    def test_longest_true_run(self) -> None:
        self.assertEqual(_longest_true_run(np.array([False, True, True, False, True])), 2)
        self.assertEqual(_longest_true_run(np.zeros(4, dtype=bool)), 0)


if __name__ == "__main__":
    unittest.main()
