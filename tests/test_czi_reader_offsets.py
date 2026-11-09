"""Pixel-correctness tests for the CZI subblock reader.

Added after `parse_subblock` was found to compute `data_offset` 16 bytes too
late, which shifted every decoded plane by 8 uint16 pixels. Nothing in the
suite caught it: `test_io_multiplane` exercises the multiplane guard with
synthetic dicts whose `data_offset` is 0, so the arithmetic was untested.

Two layers here:

* `SubblockOffsetTests` builds a synthetic ZISRAW subblock payload and checks
  the offset arithmetic directly. Runs anywhere, no data needed.
* `ReaderMatchesReferenceTests` decodes real planes and compares them to
  aicspylibczi (an independent implementation wrapping Zeiss's libCZI). Skips
  when the acquisition volume is not mounted or aicspylibczi is absent.
"""
from __future__ import annotations

import csv
import io
import struct
import unittest
from pathlib import Path

from czi_foci import czi_reader

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_directory_entry(dimensions: list[tuple[str, int, int, int]]) -> bytes:
    """Minimal valid DirectoryEntryDV: 32-byte head + 20 bytes per dimension."""
    head = bytearray(32)
    head[0:2] = b"DV"
    struct.pack_into("<i", head, 2, 1)              # pixel_type 1 -> Gray16/uint16
    struct.pack_into("<q", head, 6, 0)              # file_position
    struct.pack_into("<i", head, 14, 0)             # file_part
    struct.pack_into("<i", head, 18, 0)             # compression 0 -> uncompressed
    struct.pack_into("<i", head, 28, len(dimensions))
    body = b"".join(
        name.encode("ascii").ljust(4, b"\0") + struct.pack("<iifi", start, size, 0.0, stored)
        for name, start, size, stored in dimensions
    )
    return bytes(head) + body


class SubblockOffsetTests(unittest.TestCase):
    """data_offset must point at the first pixel byte, not past it."""

    def _parse(self, metadata_size: int, attachment_size: int, dimensions):
        data_size = 32
        payload = struct.pack("<IIQ", metadata_size, attachment_size, data_size)
        entry = build_directory_entry(dimensions)
        payload += entry
        # pad the leading block out to 256 bytes, as the writer does
        payload = payload.ljust(256, b"\0")
        payload += b"M" * metadata_size
        payload += b"P" * data_size
        payload += b"A" * attachment_size

        handle = io.BytesIO(b"\0" * 32 + payload)   # 32 bytes standing in for the segment header
        segment = czi_reader.Segment(
            offset=32 - czi_reader.SEGMENT_HEADER.size,
            identifier="ZISRAWSUBBLOCK",
            allocated_size=len(payload),
            used_size=len(payload),
        )
        return czi_reader.parse_subblock(handle, segment, include_stats=False), handle

    def test_data_offset_lands_on_the_first_pixel_byte(self) -> None:
        dims = [("X", 0, 4, 4), ("Y", 0, 4, 4), ("C", 0, 1, 1)]
        result, handle = self._parse(metadata_size=100, attachment_size=0, dimensions=dims)
        handle.seek(result["data_offset"])
        self.assertEqual(handle.read(result["data_size"]), b"P" * 32)

    def test_attachment_does_not_shift_the_data_offset(self) -> None:
        """The attachment trails the pixel data, so it must not move data_offset."""
        dims = [("X", 0, 4, 4), ("Y", 0, 4, 4), ("C", 0, 1, 1)]
        without, _ = self._parse(metadata_size=64, attachment_size=0, dimensions=dims)
        with_attachment, handle = self._parse(metadata_size=64, attachment_size=48, dimensions=dims)
        self.assertEqual(without["data_offset"], with_attachment["data_offset"])
        handle.seek(with_attachment["data_offset"])
        self.assertEqual(handle.read(with_attachment["data_size"]), b"P" * 32)


def find_sample_czi(limit: int = 3) -> list[Path]:
    sheet = REPO_ROOT / "sample_sheet.csv"
    if not sheet.exists():
        return []
    found: list[Path] = []
    with sheet.open(newline="") as fh:
        for row in csv.DictReader(fh):
            candidate = Path(row.get("source_path", ""))
            if candidate.is_file():
                found.append(candidate)
                if len(found) >= limit:
                    break
    return found


class ReaderMatchesReferenceTests(unittest.TestCase):
    """Decoded pixels must match an independent CZI implementation."""

    def setUp(self) -> None:
        try:
            from aicspylibczi import CziFile  # noqa: F401
        except ImportError:
            self.skipTest("aicspylibczi not installed")
        self.samples = find_sample_czi()
        if not self.samples:
            self.skipTest("no CZI from sample_sheet.csv is reachable (volume not mounted?)")

    def test_every_channel_matches_aicspylibczi(self) -> None:
        import numpy as np
        from aicspylibczi import CziFile

        for path in self.samples:
            inspection = czi_reader.inspect(
                path, include_stats=False, preview_dir=None, plane_dir=None
            )
            reference = CziFile(str(path))
            with self.subTest(czi=path.name):
                for block in inspection["subblocks"]:
                    dims = block["dimensions"]
                    if block["compression_code"] != 0:
                        continue
                    channel = int(dims.get("C", {}).get("start", 0))
                    z = int(dims.get("Z", {}).get("start", 0))
                    height = int(dims["Y"]["stored_size"])
                    width = int(dims["X"]["stored_size"])

                    with path.open("rb") as handle:
                        handle.seek(block["data_offset"])
                        raw = czi_reader.read_exact(handle, block["data_size"])
                    dtype = np.dtype(block["dtype"]).newbyteorder("<")
                    mine = np.frombuffer(raw, dtype=dtype).reshape((height, width))

                    kwargs = {"C": channel}
                    if "Z" in dims:
                        kwargs["Z"] = z
                    theirs = np.squeeze(reference.read_image(**kwargs)[0])

                    self.assertTrue(
                        np.array_equal(mine, theirs),
                        f"{path.name} C={channel} Z={z} differs from aicspylibczi",
                    )


if __name__ == "__main__":
    unittest.main()
