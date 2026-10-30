import tempfile
import unittest
from pathlib import Path

import pandas as pd

from synthshift.data.hhar import audit_csv, infer_timestamp_scale, validate_columns


def _frame(device: str = "nexus4_1", rate_hz: int = 100, samples: int = 20) -> pd.DataFrame:
    step_ns = int(1_000_000_000 / rate_hz)
    return pd.DataFrame(
        {
            "Arrival_Time": [index * step_ns for index in range(samples)],
            "Creation_Time": [index * step_ns for index in range(samples)],
            "x": [0.0] * samples,
            "y": [0.0] * samples,
            "z": [9.81] * samples,
            "User": ["a"] * samples,
            "Model": ["nexus4"] * samples,
            "Device": [device] * samples,
            "gt": ["walk"] * samples,
        }
    )


class HharAuditTest(unittest.TestCase):
    def test_infers_nanosecond_rate(self) -> None:
        differences = _frame()["Creation_Time"].diff().dropna().to_numpy()
        unit, rate = infer_timestamp_scale(differences) or (None, None)
        self.assertEqual(unit, "nanoseconds")
        self.assertAlmostEqual(rate, 100.0)

    def test_rejects_missing_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "Arrival_Time"):
            validate_columns(["Creation_Time", "x", "y", "z"])

    def test_audit_is_chunk_boundary_safe(self) -> None:
        frame = pd.concat([_frame(samples=12), _frame(device="s3_1", rate_hz=50, samples=12)])
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "Phones_accelerometer.csv"
            frame.to_csv(csv_path, index=False)
            report = audit_csv(csv_path, chunksize=7)

        self.assertEqual(report["rows"], 24)
        self.assertEqual(set(report["devices"]), {"nexus4_1", "s3_1"})
        self.assertEqual(
            report["timing_by_device"]["nexus4_1"]["median_effective_rate_hz_guess"], 100.0
        )
        self.assertEqual(report["timing_by_device"]["s3_1"]["median_effective_rate_hz_guess"], 50.0)
        self.assertIn("a|nexus4_1|walk", report["rows_by_user_device_label"])


if __name__ == "__main__":
    unittest.main()
