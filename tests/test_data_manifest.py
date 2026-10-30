import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from synthshift.data.manifest import build_manifest


class DataManifestTest(unittest.TestCase):
    def test_manifest_records_hash_and_zip_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "sample.zip"
            with ZipFile(archive, "w") as bundle:
                bundle.writestr("sample/data.csv", "x,y\n1,2\n")

            manifest = build_manifest(archive, "https://example.test/sample.zip")

        self.assertEqual(manifest["zip_integrity"], "passed")
        self.assertEqual(len(manifest["sha256"]), 64)
        self.assertEqual(manifest["members"][0]["path"], "sample/data.csv")

    def test_incomplete_archive_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "partial.zip"
            archive.write_bytes(b"not a complete zip")
            with self.assertRaisesRegex(ValueError, "complete ZIP"):
                build_manifest(archive, "https://example.test/partial.zip")


if __name__ == "__main__":
    unittest.main()
