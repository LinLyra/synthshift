import tempfile
import unittest
from pathlib import Path

import pandas as pd

from synthshift.data.shard import shard_csv


class ShardTest(unittest.TestCase):
    def test_shards_across_chunk_boundaries_without_duplicate_headers(self) -> None:
        frame = pd.DataFrame(
            {
                "Arrival_Time": [1, 2, 3],
                "Creation_Time": [1, 2, 3],
                "x": [0.0] * 3,
                "y": [0.0] * 3,
                "z": [0.0] * 3,
                "User": ["a", "b", "a"],
                "Device": ["phone_1"] * 3,
                "gt": ["walk"] * 3,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.csv"
            frame.to_csv(source, index=False)
            paths = shard_csv(source, root / "shards", chunksize=2)
            shard = pd.read_csv(root / "shards" / "a__phone_1.csv")
        self.assertEqual(len(paths), 2)
        self.assertEqual(len(shard), 2)

    def test_refuses_to_append_to_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "shards"
            output.mkdir()
            (output / "existing.csv").write_text("already here")
            with self.assertRaises(FileExistsError):
                shard_csv(Path(directory) / "raw.csv", output)


if __name__ == "__main__":
    unittest.main()
