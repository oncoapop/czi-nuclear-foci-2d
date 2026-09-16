from __future__ import annotations

import unittest
from pathlib import Path

from czi_foci.io import _reject_multiplane


def subblock(channel: int, z: int | None = None) -> dict:
    dims = {"C": {"start": channel}, "X": {"stored_size": 4}, "Y": {"stored_size": 4}}
    if z is not None:
        dims["Z"] = {"start": z}
    return {"dimensions": dims, "compression_code": 0, "dtype": "uint8",
            "data_offset": 0, "data_size": 16}


class MultiplaneGuardTests(unittest.TestCase):
    def test_single_plane_per_channel_is_accepted(self) -> None:
        _reject_multiplane(Path("x.czi"), [subblock(c) for c in (0, 1, 2)])

    def test_z_stack_is_rejected_rather_than_silently_collapsed(self) -> None:
        blocks = [subblock(c, z) for z in range(19) for c in (0, 1, 2)]
        with self.assertRaises(ValueError) as ctx:
            _reject_multiplane(Path("stack.czi"), blocks)
        message = str(ctx.exception)
        self.assertIn("57 subblocks across 3 channels", message)
        self.assertIn("19 planes per channel", message)
        self.assertIn("Z=19", message)

    def test_channel_index_falls_back_to_position_when_absent(self) -> None:
        blocks = [{"dimensions": {"X": {"stored_size": 4}, "Y": {"stored_size": 4}},
                   "compression_code": 0, "dtype": "uint8",
                   "data_offset": 0, "data_size": 16} for _ in range(3)]
        _reject_multiplane(Path("x.czi"), blocks)   # distinct positions, so distinct channels


if __name__ == "__main__":
    unittest.main()
