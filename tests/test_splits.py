import json
import tempfile
import unittest
from pathlib import Path

from synthshift.data.splits import load_and_validate_split


def _config() -> dict:
    return {
        "all_users": ["a", "b"],
        "folds": [
            {
                "name": "one",
                "train_users": ["a"],
                "validation_users": [],
                "test_users": ["b"],
                "target_calibration_users": ["a"],
            },
            {
                "name": "two",
                "train_users": ["b"],
                "validation_users": [],
                "test_users": ["a"],
                "target_calibration_users": ["b"],
            },
        ],
    }


class SplitValidationTest(unittest.TestCase):
    def _write(self, root: Path, config: dict) -> Path:
        path = root / "split.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_accepts_disjoint_outer_folds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            validated = load_and_validate_split(self._write(Path(directory), _config()))
        self.assertEqual(len(validated["sha256"]), 64)

    def test_rejects_partition_leakage(self) -> None:
        config = _config()
        config["folds"][0]["validation_users"] = ["a"]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(Path(directory), config)
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_and_validate_split(path)

    def test_rejects_calibration_from_test_user(self) -> None:
        config = _config()
        config["folds"][0]["target_calibration_users"] = ["b"]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(Path(directory), config)
            with self.assertRaisesRegex(ValueError, "calibration"):
                load_and_validate_split(path)


if __name__ == "__main__":
    unittest.main()
