"""File and metadata helpers for CZI foci analysis."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import czi_reader
from .config import AnalysisConfig


@dataclass(frozen=True)
class SampleMetadata:
    sample_id: str
    file_name: str
    source_path: str
    condition: str
    condition_code: str
    timepoint_hr: str
    replicate: str
    acquisition_date_folder: str


def pixel_size_um(metadata: dict) -> tuple[float, float]:
    values: dict[str, float] = {}
    for item in metadata.get("scaling") or []:
        if item.get("Id") in {"X", "Y"} and item.get("Value"):
            values[item["Id"]] = float(item["Value"]) * 1_000_000.0
    return values.get("X", float("nan")), values.get("Y", float("nan"))


def channel_metadata(inspection: dict, index: int) -> str:
    channels = inspection["metadata"].get("channels") or []
    if index >= len(channels):
        return f"C{index}"
    channel = channels[index]
    parts = [f"C{index}", channel.get("Name", ""), channel.get("Fluor", "")]
    ex = channel.get("ExcitationWavelength")
    em = channel.get("EmissionWavelength")
    if ex and em:
        parts.append(f"Ex {ex} / Em {em} nm")
    return " | ".join(p for p in parts if p)


def inspect_czi(path: Path) -> dict:
    return czi_reader.inspect(path, include_stats=False, preview_dir=None, plane_dir=None)


def _reject_multiplane(path: Path, subblocks: list[dict]) -> None:
    """Refuse files holding more than one plane per channel.

    This reader keeps a single 2-D plane per channel. A Z-stack, time series,
    tiled mosaic or multi-scene file has several subblocks per channel, and
    without this check the last one silently wins - a 19-plane Z-stack collapses
    to its final, out-of-focus plane with no error raised.
    """
    per_channel: dict[int, int] = {}
    extra: dict[str, set[int]] = {}
    for index, block in enumerate(subblocks):
        dims = block["dimensions"]
        channel = int(dims.get("C", {}).get("start", index))
        per_channel[channel] = per_channel.get(channel, 0) + 1
        for name in ("Z", "T", "S", "M"):
            if name in dims:
                extra.setdefault(name, set()).add(int(dims[name].get("start", 0)))
    crowded = [n for n in per_channel.values() if n > 1]
    if not crowded:
        return
    spread = ", ".join(f"{k}={len(v)}" for k, v in sorted(extra.items()) if len(v) > 1)
    raise ValueError(
        f"Multi-plane CZI is not supported by the 2-D reader: {path}\n"
        f"  {len(subblocks)} subblocks across {len(per_channel)} channels "
        f"({max(crowded)} planes per channel; {spread or 'unidentified extra dimension'}).\n"
        f"  Project to a single plane first, or use the 3-D analysis mode."
    )


def read_channel_arrays(path: Path, inspection: dict) -> dict[int, np.ndarray]:
    _reject_multiplane(path, inspection["subblocks"])
    arrays: dict[int, np.ndarray] = {}
    with path.open("rb") as handle:
        for index, block in enumerate(inspection["subblocks"]):
            if block["compression_code"] != 0:
                raise ValueError(f"Compressed CZI subblock is not supported: {path}")
            if block["dtype"] not in {"uint8", "uint16"}:
                raise ValueError(f"Expected Gray8/Gray16 channel data, found {block['dtype']}: {path}")
            dims = block["dimensions"]
            channel = int(dims.get("C", {}).get("start", index))
            width = int(dims["X"]["stored_size"])
            height = int(dims["Y"]["stored_size"])
            dtype = np.dtype(block["dtype"]).newbyteorder("<")
            handle.seek(block["data_offset"])
            raw = czi_reader.read_exact(handle, block["data_size"])
            arrays[channel] = np.frombuffer(raw, dtype=dtype).reshape((height, width))
    return arrays


def load_condition_map(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    if "condition_code" not in rows[0] or "condition" not in rows[0]:
        raise ValueError("Condition map must include condition_code and condition columns")
    return {row["condition_code"]: row for row in rows}


def parse_sample_metadata(path: Path, config: AnalysisConfig, condition_map: dict[str, dict[str, str]]) -> SampleMetadata:
    match = re.match(config.filename.regex, path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Filename does not match configured regex: {path.name}")
    condition_code = match.group(config.filename.condition_group).strip()
    mapped = condition_map.get(condition_code, {})
    condition = mapped.get("condition", condition_code).strip()
    timepoint = match.group(config.filename.timepoint_group).strip()
    replicate = match.group(config.filename.replicate_group).strip().zfill(2)
    date_folder = path.parent.name if re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.parent.name) else ""
    safe_condition = re.sub(r"[^A-Za-z0-9]+", "_", condition).strip("_") or "conditionUNK"
    safe_experiment = re.sub(r"[^A-Za-z0-9]+", "_", config.experiment_name).strip("_") or "experiment"
    sample_id = f"{safe_experiment}_{date_folder or 'dateUNK'}_{timepoint}hr_{safe_condition}_rep{replicate}"
    return SampleMetadata(
        sample_id=sample_id,
        file_name=path.name,
        source_path=str(path),
        condition=condition,
        condition_code=condition_code,
        timepoint_hr=timepoint,
        replicate=replicate,
        acquisition_date_folder=date_folder,
    )


def selected_files_from_root(root: Path, config: AnalysisConfig) -> list[Path]:
    pattern = re.compile(config.filename.regex, re.IGNORECASE)
    return sorted(path for path in root.rglob("*.czi") if pattern.match(path.name))


def selected_files_from_manifest(manifest: Path) -> list[Path]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "source_path" not in rows[0]:
        raise ValueError("Manifest must contain a source_path column")
    return [Path(row["source_path"]) for row in rows]
