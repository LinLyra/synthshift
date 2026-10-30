import unittest

import numpy as np

from synthshift.eval.residual_audit import _correlation, _profile_map


class ResidualAuditTest(unittest.TestCase):
    def test_profile_map_uses_users_and_device(self) -> None:
        profiles = [
            {"users": ["a", "h"], "device": "phone", "gain": 1.0},
            {"users": ["b", "g"], "device": "phone", "gain": 1.1},
        ]
        mapped = _profile_map(profiles)
        self.assertEqual(mapped[(("a", "h"), "phone")]["gain"], 1.0)
        self.assertEqual(mapped[(("b", "g"), "phone")]["gain"], 1.1)

    def test_profile_map_rejects_duplicate_key(self) -> None:
        profile = {"users": ["a", "h"], "device": "phone"}
        with self.assertRaises(ValueError):
            _profile_map([profile, profile])

    def test_correlation_reports_known_order(self) -> None:
        result = _correlation(np.arange(5.0), np.arange(5.0))
        self.assertAlmostEqual(result["pearson_r"], 1.0)
        self.assertAlmostEqual(result["spearman_r"], 1.0)


if __name__ == "__main__":
    unittest.main()
