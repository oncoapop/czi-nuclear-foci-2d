#!/usr/bin/env python3
"""Create a labelled CZI review PDF from a small sample set."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


DEFAULT_ROOT = Path("/Volumes/Backup/TS runs/Time Series III (quarter dilution)/2026-05-15")
CONDITION_ORDER = [
    "Aqueous",
    "DMSO",
    "NAH2PO4",
    "CX-5461",
    "Cisplatin",
    "Etoposide",
    "PDS",
    "Palbociclib",
]


def load_inspector():
    script = Path(__file__).with_name("inspect_czi.py")
    spec = importlib.util.spec_from_file_location("inspect_czi", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def parse_condition(path: Path) -> tuple[str, str]:
    patterns = [
        r"^(?P<timepoint>\d+)\s*hr\s+20x\s+(?P<condition>.+)-(?P<replicate>\d+)\.czi$",
        r"^(?P<timepoint>\d+)\s*hr\s+(?P<condition>.+?)\s+20x\s*-?\s*(?P<replicate>\d+)\.czi$",
    ]
    for pattern in patterns:
        match = re.match(pattern, path.name, re.IGNORECASE)
        if match:
            return match.group("condition").strip(), match.group("replicate").zfill(2)
    return "", ""


def selected_files(root: Path, timepoint_hr: str, replicate: str) -> list[Path]:
    files = sorted(root.rglob(f"{timepoint_hr} hr *20x*{replicate}.czi"))
    by_condition = {parse_condition(path)[0]: path for path in files}
    return [by_condition[name] for name in CONDITION_ORDER if name in by_condition]


def read_channel_arrays(czi_path: Path, inspection: dict, inspector) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    with czi_path.open("rb") as handle:
        for block in inspection["subblocks"]:
            if block["compression_code"] != 0:
                raise ValueError(f"Compressed CZI subblock is not supported: {czi_path}")
            if block["dtype"] != "uint8":
                raise ValueError(f"Expected uint8 channel data, found {block['dtype']}: {czi_path}")
            dims = block["dimensions"]
            width = dims["X"]["stored_size"]
            height = dims["Y"]["stored_size"]
            handle.seek(block["data_offset"])
            raw = inspector.read_exact(handle, block["data_size"])
            arrays.append(np.frombuffer(raw, dtype=np.uint8).reshape((height, width)))
    return arrays


def stretch_for_display(array: np.ndarray) -> np.ndarray:
    low, high = np.percentile(array, [1, 99.8])
    if high <= low:
        low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.zeros_like(array, dtype=np.uint8)
    scaled = (array.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def image_reader(array: np.ndarray, mode: str = "L") -> ImageReader:
    image = Image.fromarray(array, mode=mode)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return ImageReader(buffer)


def channel_name(metadata: dict, index: int) -> str:
    channels = metadata.get("channels") or []
    if index < len(channels):
        channel = channels[index]
        name = channel.get("Name") or f"Channel {index}"
        fluor = channel.get("Fluor")
        excitation = channel.get("ExcitationWavelength")
        emission = channel.get("EmissionWavelength")
        details = []
        if fluor:
            details.append(fluor)
        if excitation and emission:
            details.append(f"Ex {excitation} / Em {emission} nm")
        return f"C{index}: {name}" + (f" ({'; '.join(details)})" if details else "")
    return f"C{index}"


def draw_wrapped(c: canvas.Canvas, text: str, x: float, y: float, max_chars: int, leading: float = 10) -> float:
    for line in wrap_text(text, max_chars):
        c.drawString(x, y, line)
        y -= leading
    return y


def selected_objective(metadata: dict) -> str:
    objectives = metadata.get("objectives") or []
    for objective in objectives:
        if str(objective.get("Id", "")).startswith("Objective:"):
            name = objective.get("Name", "")
            mag = objective.get("NominalMagnification", "")
            na = objective.get("LensNA", "")
            return f"{name}; nominal {mag}x; NA {na}".strip("; ")
    return ""


def pixel_size_um(metadata: dict) -> str:
    values = {}
    for item in metadata.get("scaling") or []:
        if item.get("Id") in {"X", "Y"} and item.get("Value"):
            values[item["Id"]] = float(item["Value"]) * 1_000_000.0
    if {"X", "Y"} <= values.keys():
        return f"X={values['X']:.6f} um/pixel, Y={values['Y']:.6f} um/pixel"
    return ""


def draw_summary_page(c: canvas.Canvas, files: list[Path], title: str, scope: str) -> None:
    width, height = landscape(A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(36, height - 40, title)
    c.setFont("Helvetica", 10)
    y = height - 68
    lines = [
        f"Scope: {scope}.",
        "Source: filenames plus CZI embedded metadata; source files read-only.",
        "Display: per-panel 1-99.8 percentile stretch for visual checking; raw 8-bit statistics are printed per channel.",
        "Channel map from metadata: C0 RhReX/Rhodamine Red-X, C1 AF488/Alexa Fluor 488, C2 DAPI.",
        f"Files sampled: {len(files)}.",
    ]
    for line in lines:
        y = draw_wrapped(c, line, 36, y, 125, 13)
    y -= 8
    c.setFont("Helvetica-Bold", 10)
    c.drawString(36, y, "Files")
    y -= 14
    c.setFont("Helvetica", 9)
    for path in files:
        y = draw_wrapped(c, path.name, 48, y, 120, 11)
    c.showPage()


def draw_file_page(c: canvas.Canvas, path: Path, inspection: dict, arrays: list[np.ndarray]) -> dict[str, str]:
    width, height = landscape(A4)
    metadata = inspection["metadata"]
    image_meta = metadata["image"]
    condition, replicate = parse_condition(path)

    c.setFont("Helvetica-Bold", 13)
    c.drawString(36, height - 34, f"{condition} | replicate {replicate}")
    c.setFont("Helvetica", 8.5)
    y = height - 52
    y = draw_wrapped(c, f"File: {path.name}", 36, y, 126, 10)
    y = draw_wrapped(c, f"Path: {path}", 36, y, 126, 10)

    c.setFont("Helvetica", 8)
    objective = selected_objective(metadata)
    px_size = pixel_size_um(metadata)
    meta_line = (
        f"Acquired {image_meta.get('AcquisitionDateAndTime', '')}; "
        f"{image_meta.get('SizeX')}x{image_meta.get('SizeY')}; "
        f"Z={image_meta.get('SizeZ')}; C={image_meta.get('SizeC')}; T={image_meta.get('SizeT')}; "
        f"{image_meta.get('PixelType')} {image_meta.get('ComponentBitCount')}-bit; {px_size}; {objective}"
    )
    y = draw_wrapped(c, meta_line, 36, y - 2, 132, 9)

    stretched = [stretch_for_display(array) for array in arrays]
    composite = np.zeros((*stretched[0].shape, 3), dtype=np.uint8)
    composite[:, :, 0] = stretched[0]
    composite[:, :, 1] = stretched[1]
    composite[:, :, 2] = stretched[2]

    panels = [("RGB composite", composite, "RGB")]
    panels.extend((channel_name(metadata, idx), stretched[idx], "L") for idx in range(len(stretched)))

    panel_w = 180
    panel_h = 180
    gap = 12
    x0 = 36
    image_y = 210
    c.setFont("Helvetica-Bold", 8)
    for idx, (label, array, mode) in enumerate(panels):
        x = x0 + idx * (panel_w + gap)
        c.drawImage(image_reader(array, mode), x, image_y, panel_w, panel_h, preserveAspectRatio=True, anchor="c")
        draw_wrapped(c, label, x, image_y - 10, 30, 9)

    table_y = 182
    c.setFont("Helvetica-Bold", 8)
    c.drawString(36, table_y, "Raw channel statistics")
    table_y -= 11
    c.setFont("Helvetica", 7.5)
    for idx, (array, block) in enumerate(zip(arrays, inspection["subblocks"])):
        saturated = int((array == 255).sum())
        zeros = int((array == 0).sum())
        n = int(array.size)
        stats = block.get("statistics", {})
        line = (
            f"C{idx}: {channel_name(metadata, idx)} | "
            f"mean={stats.get('mean', float(array.mean())):.3f}; "
            f"min={array.min()}; max={array.max()}; "
            f"zero={zeros}/{n} ({zeros / n:.3%}); saturated={saturated}/{n} ({saturated / n:.3%})"
        )
        table_y = draw_wrapped(c, line, 42, table_y, 145, 9)

    c.setFont("Helvetica", 7)
    c.drawRightString(width - 36, 20, "Generated for visual QA; quantitative analysis should use raw CZI/TIFF intensities.")
    c.showPage()

    return {
        "file_name": path.name,
        "source_path": str(path),
        "condition": condition,
        "timepoint_hr": "2",
        "replicate": replicate,
        "acquisition_datetime": image_meta.get("AcquisitionDateAndTime", ""),
        "size_x": image_meta.get("SizeX", ""),
        "size_y": image_meta.get("SizeY", ""),
        "size_c": image_meta.get("SizeC", ""),
        "pixel_type": image_meta.get("PixelType", ""),
        "component_bit_count": image_meta.get("ComponentBitCount", ""),
        "pixel_size": pixel_size_um(metadata),
        "objective": selected_objective(metadata),
        "channels": " | ".join(channel_name(metadata, idx) for idx in range(len(arrays))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--timepoint-hr", default="2")
    parser.add_argument("--replicate", default="01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--title", default="Time Series III Review Sample")
    parser.add_argument("--scope", default="Time Series III (quarter dilution), 2 hr, replicate 01")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for output in (args.output, args.manifest_output):
        if output.exists() and not args.overwrite:
            parser.error(f"Refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    files = selected_files(args.root, args.timepoint_hr, args.replicate)
    if not files:
        parser.error("No matching CZI files found")

    inspector = load_inspector()
    pdf = canvas.Canvas(str(args.output), pagesize=landscape(A4))
    draw_summary_page(pdf, files, args.title, args.scope)

    manifest_rows = []
    for path in files:
        inspection = inspector.inspect(path, True, None)
        arrays = read_channel_arrays(path, inspection, inspector)
        manifest_rows.append(draw_file_page(pdf, path, inspection, arrays))
    pdf.save()

    with args.manifest_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {args.output}")
    print(f"Wrote {args.manifest_output}")
    print(f"Sampled {len(files)} CZI files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
