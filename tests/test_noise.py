import tempfile
import unittest
from pathlib import Path

import numpy as np

from synthshift.data.still import still_identity_report
from synthshift.devices.noise import (
    added_noise_scale,
    difference_sigma,
    low_motion_differences,
)


class NoiseTest(unittest.TestCase):
    def test_difference_sigma_recovers_white_noise(self) -> None:
        generator = np.random.default_rng(0)
        sigma = np.array([0.02, 0.02, 0.03, 0.001, 0.001, 0.002])
        values = np.zeros((20_000, 6))
        values[:, 2] = 9.80665
        values += generator.normal(0.0, sigma, size=values.shape)
        estimated = difference_sigma(low_motion_differences(values))
        np.testing.assert_allclose(estimated, sigma, rtol=0.2, atol=0.004)

    def test_motion_contributes_no_differences(self) -> None:
        values = np.zeros((80, 6))
        values[:, 2] = 9.80665
        values[:, 3] = 2.0
        self.assertEqual(len(low_motion_differences(values)), 0)

    def test_added_scale_cannot_make_a_channel_quieter(self) -> None:
        scale = added_noise_scale(
            np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1]),
            np.array([0.1, 0.4, 0.2, 0.1, 0.0, 0.2]),
        )
        expected = [0.0, np.sqrt(0.16 - 0.04), 0.0, 0.0, 0.0, np.sqrt(0.04 - 0.01)]
        np.testing.assert_allclose(scale, expected)

    def test_still_names_do_not_authorize_instance_noise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "Phoneonback" / "nexus4_2" / "LG-Nexus4.csv"
            path.parent.mkdir(parents=True)
            path.write_text("Index,Arrival_Time,Creation_Time,x,y,z\n")
            other = root / "Phoneonback" / "it-151" / "LG-Nexus4.csv"
            other.parent.mkdir(parents=True)
            other.write_text("Index,Arrival_Time,Creation_Time,x,y,z\n")
            report = still_identity_report(root, ["nexus4_1", "nexus4_2", "s3_1"])
        self.assertEqual(report["shared_names"], ["nexus4_2"])
        self.assertFalse(report["use_for_instance_noise"])


if __name__ == "__main__":
    unittest.main()
