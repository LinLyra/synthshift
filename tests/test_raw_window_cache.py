import tempfile
import unittest
from pathlib import Path

import numpy as np

from synthshift.data.raw_window_cache import build_raw_window_cache, sha256_file


class RawWindowCacheTest(unittest.TestCase):
    def test_cache_preserves_feature_row_order_and_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segment_root = root / "segments"
            segment_root.mkdir()
            first = np.arange(36, dtype=np.float32).reshape(6, 6)
            second = np.arange(60, 96, dtype=np.float32).reshape(6, 6)
            np.savez(segment_root / "a.npz", values=first)
            np.savez(segment_root / "b.npz", values=second)
            features = root / "features.npz"
            np.savez(
                features,
                features=np.zeros((3, 1), dtype=np.float32),
                segment_files=np.asarray(["b.npz", "a.npz", "b.npz"]),
                offsets=np.asarray([1, 2, 0]),
            )
            output = root / "windows.npy"
            manifest = root / "manifest.json"
            result = build_raw_window_cache(
                features, segment_root, output, manifest, window_samples=2
            )
            cached = np.load(output)
            np.testing.assert_array_equal(cached[0], second[1:3])
            np.testing.assert_array_equal(cached[1], first[2:4])
            np.testing.assert_array_equal(cached[2], second[0:2])
            self.assertEqual(result["shape"], [3, 2, 6])
            self.assertEqual(result["cache_sha256"], sha256_file(output))


if __name__ == "__main__":
    unittest.main()
