import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from czi_foci.focus_qc import _load_approved_mask


class ApprovedMaskTests(unittest.TestCase):
    def test_loads_matching_integer_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.tif"
            Image.fromarray(np.array([[0, 1], [2, 2]], dtype=np.uint16)).save(path)
            result = _load_approved_mask(path, (2, 2))
            self.assertEqual(result.dtype, np.uint16)
            self.assertEqual(int(result.max()), 2)

    def test_rejects_wrong_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.tif"
            Image.fromarray(np.zeros((2, 2), dtype=np.uint16)).save(path)
            with self.assertRaisesRegex(ValueError, "does not match"):
                _load_approved_mask(path, (3, 2))


if __name__ == "__main__":
    unittest.main()
