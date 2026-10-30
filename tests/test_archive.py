import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from synthshift.data.archive import safe_extract_zip


class SafeExtractionTest(unittest.TestCase):
    def test_extracts_regular_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "data.zip"
            with ZipFile(archive, "w") as bundle:
                bundle.writestr("nested/data.csv", "x,y\n1,2\n")

            extracted = safe_extract_zip(archive, root / "output")

            self.assertEqual(len(extracted), 1)
            self.assertEqual(extracted[0].read_text(), "x,y\n1,2\n")

    def test_rejects_parent_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "malicious.zip"
            with ZipFile(archive, "w") as bundle:
                bundle.writestr("../escaped.txt", "must not be written")

            with self.assertRaisesRegex(ValueError, "path traversal"):
                safe_extract_zip(archive, root / "output")
            self.assertFalse((root / "escaped.txt").exists())


if __name__ == "__main__":
    unittest.main()
