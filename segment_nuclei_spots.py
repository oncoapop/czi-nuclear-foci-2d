#!/usr/bin/env python3
"""Segment DAPI nuclei and AF488 nuclear spots from CZI files."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


DEFAULT_MANIFEST = Path(
    "/Users/dyap/Documents/Tasks workflow/czi-analysis/output/review/"
    "time_series_III_2hr_rep01_review_manifest.csv"
)
DEFAULT_ROOT = Path("/Volumes/Backup/TS runs/Time Series III (quarter dilution)")
DEFAULT_OUTPUT = Path(
    "/Users/dyap/Documents/Tasks workflow/czi-analysis/output/segmentation/"
    "time_series_III_2hr_rep01"
)


@dataclass(frozen=True)
class Parameters:
    dapi_channel_index: int = 2
    af488_channel_index: int = 1
    nuclei_gaussian_sigma_px: float = 1.2
    nuclei_min_area_um2: float = 35.0
    nuclei_hole_area_um2: float = 20.0
    nuclei_watershed_min_distance_px: int = 10
    spot_tophat_radius_px: int = 4
    spot_min_area_px: int = 3
    spot_threshold_mad_multiplier: float = 4.0
    spot_min_intensity_above_local_background: float = 8.0
    qc_display_percentile_low: float = 1.0
    qc_display_percentile_high: float = 99.8


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


def parse_condition(path: Path) -> tuple[str, str, str]:
    patterns = [
        r"^(?P<timepoint>\d+)\s*hr\s+20x\s+(?P<condition>.+)-(?P<replicate>\d+)\.czi$",
        r"^(?P<timepoint>\d+)\s*hr\s+(?P<condition>.+?)\s+20x\s*-?\s*(?P<replicate>\d+)\.czi$",
    ]
    for pattern in patterns:
        match = re.match(pattern, path.name, re.IGNORECASE)
        if match:
            return (
                match.group("condition").strip(),
                match.group("timepoint").strip(),
                match.group("replicate").zfill(2),
            )
    return "", "", ""


def series_number(path: Path) -> str:
    for part in path.parts:
        match = re.match(r"^Time Series\s+([IVX]+)", part)
        if match:
            return match.group(1)
    return "UNK"


def sample_id(path: Path) -> str:
    condition, timepoint, replicate = parse_condition(path)
    date_folder = path.parent.name if re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.parent.name) else "dateUNK"
    safe_condition = re.sub(r"[^A-Za-z0-9]+", "_", condition).strip("_")
    return f"TS{series_number(path)}_{date_folder}_{timepoint}hr_{safe_condition}_rep{replicate}"


def acquisition_date_folder(path: Path) -> str:
    return path.parent.name if re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.parent.name) else ""


def file_sort_key(path: Path) -> tuple[int, int, int, str]:
    condition, timepoint, replicate = parse_condition(path)
    condition_rank = CONDITION_ORDER.index(condition) if condition in CONDITION_ORDER else len(CONDITION_ORDER)
    return (int(timepoint or 999), int(replicate or 999), condition_rank, path.name)


def selected_files_from_manifest(manifest: Path) -> list[Path]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    paths = [Path(row["source_path"]) for row in rows]
    return sorted(paths, key=file_sort_key)


def selected_files_from_root(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.rglob("*.czi") if all(parse_condition(path))],
        key=file_sort_key,
    )


def pixel_size_um(metadata: dict) -> tuple[float, float]:
    values: dict[str, float] = {}
    for item in metadata.get("scaling") or []:
        if item.get("Id") in {"X", "Y"} and item.get("Value"):
            values[item["Id"]] = float(item["Value"]) * 1_000_000.0
    return values.get("X", float("nan")), values.get("Y", float("nan"))


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


def area_um2_to_px(area_um2: float, px_um_x: float, px_um_y: float) -> int:
    if not np.isfinite(px_um_x) or not np.isfinite(px_um_y) or px_um_x <= 0 or px_um_y <= 0:
        return int(round(area_um2))
    return max(1, int(round(area_um2 / (px_um_x * px_um_y))))


def segment_nuclei(dapi: np.ndarray, px_um_x: float, px_um_y: float, params: Parameters) -> tuple[np.ndarray, dict]:
    smoothed = filters.gaussian(dapi, sigma=params.nuclei_gaussian_sigma_px, preserve_range=True)
    threshold = float(filters.threshold_otsu(smoothed))
    min_area = area_um2_to_px(params.nuclei_min_area_um2, px_um_x, px_um_y)
    hole_area = area_um2_to_px(params.nuclei_hole_area_um2, px_um_x, px_um_y)

    binary = smoothed > threshold
    binary = morphology.remove_small_objects(binary, min_size=min_area)
    binary = morphology.remove_small_holes(binary, area_threshold=hole_area)
    binary = ndi.binary_fill_holes(binary)

    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(
        distance,
        min_distance=params.nuclei_watershed_min_distance_px,
        labels=binary,
        exclude_border=False,
    )
    markers = np.zeros(binary.shape, dtype=np.int32)
    if coords.size:
        markers[tuple(coords.T)] = np.arange(1, coords.shape[0] + 1)
    markers = measure.label(markers > 0)
    if markers.max() == 0:
        labels = measure.label(binary)
    else:
        labels = segmentation.watershed(-distance, markers, mask=binary)
    labels = morphology.remove_small_objects(labels, min_size=min_area)
    labels = measure.label(labels > 0)

    diagnostics = {
        "dapi_otsu_threshold": threshold,
        "nuclei_min_area_px": min_area,
        "nuclei_hole_area_px": hole_area,
        "initial_watershed_markers": int(markers.max()),
        "nuclei_count": int(labels.max()),
    }
    return labels.astype(np.uint16), diagnostics


def robust_threshold(values: np.ndarray, multiplier: float) -> float:
    if values.size == 0:
        return float("inf")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    mad_sigma = 1.4826 * mad
    otsu = float(filters.threshold_otsu(values)) if np.unique(values).size > 1 else median
    return max(otsu, median + multiplier * mad_sigma)


def segment_spots_in_nuclei(
    af488: np.ndarray,
    nuclei_labels: np.ndarray,
    params: Parameters,
) -> tuple[np.ndarray, dict]:
    nuclear_mask = nuclei_labels > 0
    footprint = morphology.disk(params.spot_tophat_radius_px)
    enhanced = morphology.white_tophat(af488, footprint=footprint)
    nuclear_values = enhanced[nuclear_mask]
    threshold = robust_threshold(nuclear_values, params.spot_threshold_mad_multiplier)
    local_background = filters.gaussian(af488, sigma=params.spot_tophat_radius_px * 2, preserve_range=True)
    above_background = af488.astype(np.float32) - local_background.astype(np.float32)

    binary = (
        nuclear_mask
        & (enhanced >= threshold)
        & (above_background >= params.spot_min_intensity_above_local_background)
    )
    binary = morphology.remove_small_objects(binary, min_size=params.spot_min_area_px)
    labels = measure.label(binary).astype(np.uint16)
    diagnostics = {
        "af488_tophat_threshold": float(threshold),
        "spot_min_area_px": params.spot_min_area_px,
        "spot_count": int(labels.max()),
    }
    return labels, diagnostics


def display_u8(image: np.ndarray, params: Parameters) -> np.ndarray:
    low, high = np.percentile(
        image,
        [params.qc_display_percentile_low, params.qc_display_percentile_high],
    )
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)
    return np.clip((image.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def save_label_tiff(labels: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    Image.fromarray(labels).save(path, format="TIFF")


def save_qc_overlay(
    dapi: np.ndarray,
    af488: np.ndarray,
    nuclei_labels: np.ndarray,
    spot_labels: np.ndarray,
    path: Path,
    params: Parameters,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    dapi_disp = display_u8(dapi, params)
    af488_disp = display_u8(af488, params)
    rgb = np.zeros((*dapi.shape, 3), dtype=np.uint8)
    rgb[:, :, 1] = af488_disp
    rgb[:, :, 2] = dapi_disp

    nuclei_boundary = segmentation.find_boundaries(nuclei_labels, mode="outer")
    spots_boundary = segmentation.find_boundaries(spot_labels, mode="outer")
    rgb[nuclei_boundary] = np.array([255, 255, 0], dtype=np.uint8)
    rgb[spots_boundary] = np.array([255, 0, 0], dtype=np.uint8)

    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 5, 420, 55), fill=(0, 0, 0))
    draw.text((12, 12), "Blue=DAPI/T3, Green=AF488/T2", fill=(255, 255, 255))
    draw.text((12, 31), "Yellow=nucleus boundary, Red=AF488 spot boundary", fill=(255, 255, 255))
    image.save(path)


def region_rows(
    path: Path,
    dapi: np.ndarray,
    af488: np.ndarray,
    nuclei_labels: np.ndarray,
    spot_labels: np.ndarray,
    px_um_x: float,
    px_um_y: float,
) -> tuple[list[dict], list[dict]]:
    condition, timepoint, replicate = parse_condition(path)
    sid = sample_id(path)
    area_scale = px_um_x * px_um_y if np.isfinite(px_um_x * px_um_y) else float("nan")
    spots_by_nucleus: dict[int, list] = {}

    spot_rows = []
    for spot in measure.regionprops(spot_labels, intensity_image=af488):
        coords = spot.coords
        nucleus_values = nuclei_labels[coords[:, 0], coords[:, 1]]
        nucleus_values = nucleus_values[nucleus_values > 0]
        nucleus_label = int(np.bincount(nucleus_values.astype(np.int64)).argmax()) if nucleus_values.size else 0
        spots_by_nucleus.setdefault(nucleus_label, []).append(spot)
        spot_rows.append(
            {
                "sample_id": sid,
                "file_name": path.name,
                "acquisition_date_folder": acquisition_date_folder(path),
                "condition": condition,
                "timepoint_hr": timepoint,
                "replicate": replicate,
                "nucleus_label": nucleus_label,
                "spot_label": int(spot.label),
                "spot_area_px": int(spot.area),
                "spot_area_um2": float(spot.area * area_scale),
                "spot_mean_af488_intensity": float(spot.mean_intensity),
                "spot_max_af488_intensity": float(spot.max_intensity),
                "spot_centroid_y_px": float(spot.centroid[0]),
                "spot_centroid_x_px": float(spot.centroid[1]),
            }
        )

    nucleus_rows = []
    for nucleus in measure.regionprops(nuclei_labels, intensity_image=dapi):
        label = int(nucleus.label)
        nucleus_mask = nuclei_labels == label
        spots = spots_by_nucleus.get(label, [])
        nucleus_rows.append(
            {
                "sample_id": sid,
                "file_name": path.name,
                "condition": condition,
                "timepoint_hr": timepoint,
                "replicate": replicate,
                "nucleus_label": label,
                "nucleus_area_px": int(nucleus.area),
                "nucleus_area_um2": float(nucleus.area * area_scale),
                "nucleus_mean_dapi_intensity": float(nucleus.mean_intensity),
                "nucleus_mean_af488_intensity": float(af488[nucleus_mask].mean()),
                "nucleus_centroid_y_px": float(nucleus.centroid[0]),
                "nucleus_centroid_x_px": float(nucleus.centroid[1]),
                "spots_count": len(spots),
                "spots_total_area_px": int(sum(spot.area for spot in spots)),
                "spots_total_area_um2": float(sum(spot.area for spot in spots) * area_scale),
                "nucleus_has_gt3_spots": "TRUE" if len(spots) > 3 else "FALSE",
            }
        )

    return nucleus_rows, spot_rows


def image_summary_row(
    path: Path,
    inspection: dict,
    dapi: np.ndarray,
    af488: np.ndarray,
    nuclei_rows: list[dict],
    spot_rows: list[dict],
    nuclei_diag: dict,
    spots_diag: dict,
) -> dict:
    condition, timepoint, replicate = parse_condition(path)
    nuclei_count = len(nuclei_rows)
    gt3 = sum(1 for row in nuclei_rows if row["nucleus_has_gt3_spots"] == "TRUE")
    mean_spots = float(np.mean([row["spots_count"] for row in nuclei_rows])) if nuclei_rows else 0.0
    return {
        "sample_id": sample_id(path),
        "file_name": path.name,
        "source_path": str(path),
        "acquisition_date_folder": acquisition_date_folder(path),
        "condition": condition,
        "timepoint_hr": timepoint,
        "replicate": replicate,
        "acquisition_datetime": inspection["metadata"]["image"].get("AcquisitionDateAndTime", ""),
        "nuclei_count": nuclei_count,
        "mean_nucleus_area_um2": float(np.mean([row["nucleus_area_um2"] for row in nuclei_rows])) if nuclei_rows else 0.0,
        "median_nucleus_area_um2": float(np.median([row["nucleus_area_um2"] for row in nuclei_rows])) if nuclei_rows else 0.0,
        "mean_nucleus_dapi_intensity": float(np.mean([row["nucleus_mean_dapi_intensity"] for row in nuclei_rows])) if nuclei_rows else 0.0,
        "mean_nuclear_af488_intensity": float(np.mean([row["nucleus_mean_af488_intensity"] for row in nuclei_rows])) if nuclei_rows else 0.0,
        "total_af488_spots_in_nuclei": len(spot_rows),
        "mean_af488_spots_per_nucleus": mean_spots,
        "nuclei_with_gt3_spots": gt3,
        "fraction_nuclei_with_gt3_spots": float(gt3 / nuclei_count) if nuclei_count else 0.0,
        "raw_dapi_mean_intensity": float(dapi.mean()),
        "raw_af488_mean_intensity": float(af488.mean()),
        **nuclei_diag,
        **spots_diag,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_qc_pdf(path: Path, overlay_paths: list[Path], title: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    width, height = landscape(A4)
    pdf = canvas.Canvas(str(path), pagesize=landscape(A4))
    thumb_w = 180
    thumb_h = 180
    margin_x = 36
    gap_x = 18
    gap_y = 36
    top = height - 70
    per_page = 8
    for page_start in range(0, len(overlay_paths), per_page):
        page_paths = overlay_paths[page_start : page_start + per_page]
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(36, height - 32, title)
        pdf.setFont("Helvetica", 8)
        pdf.drawString(
            36,
            height - 46,
            "Blue=DAPI/T3; green=AF488/T2; yellow=nucleus boundary; red=AF488 spot boundary.",
        )
        pdf.drawRightString(width - 36, height - 46, f"Page {page_start // per_page + 1}")
        for index, overlay in enumerate(page_paths):
            col = index % 4
            row = index // 4
            x = margin_x + col * (thumb_w + gap_x)
            y = top - row * (thumb_h + gap_y) - thumb_h
            pdf.drawImage(
                ImageReader(str(overlay)),
                x,
                y,
                thumb_w,
                thumb_h,
                preserveAspectRatio=True,
                anchor="c",
            )
            pdf.drawString(x, y - 11, overlay.stem.replace("_qc_overlay", ""))
        pdf.showPage()
    pdf.save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, help="Process all matching CZI files below this root instead of a manifest")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Process only the first N selected files")
    parser.add_argument("--qc-title", default="Time Series III Segmentation QC")
    args = parser.parse_args()

    if args.output_dir.exists():
        if not args.overwrite:
            parser.error(f"Refusing to overwrite existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    params = Parameters()
    (args.output_dir / "parameters.json").write_text(
        json.dumps(asdict(params), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    files = selected_files_from_root(args.root) if args.root else selected_files_from_manifest(args.manifest)
    if args.limit:
        files = files[: args.limit]
    if not files:
        parser.error("No CZI files selected")

    inspector = load_inspector()
    image_rows: list[dict] = []
    nucleus_rows_all: list[dict] = []
    spot_rows_all: list[dict] = []
    overlay_paths: list[Path] = []

    for path in files:
        sid = sample_id(path)
        inspection = inspector.inspect(path, True, None)
        arrays = read_channel_arrays(path, inspection, inspector)
        dapi = arrays[params.dapi_channel_index]
        af488 = arrays[params.af488_channel_index]
        px_um_x, px_um_y = pixel_size_um(inspection["metadata"])

        nuclei_labels, nuclei_diag = segment_nuclei(dapi, px_um_x, px_um_y, params)
        spot_labels, spots_diag = segment_spots_in_nuclei(af488, nuclei_labels, params)
        nucleus_rows, spot_rows = region_rows(path, dapi, af488, nuclei_labels, spot_labels, px_um_x, px_um_y)

        save_label_tiff(nuclei_labels, args.output_dir / "masks" / f"{sid}_nuclei_labels.tif")
        save_label_tiff(spot_labels, args.output_dir / "masks" / f"{sid}_af488_spot_labels.tif")
        overlay_path = args.output_dir / "qc_overlays" / f"{sid}_qc_overlay.png"
        save_qc_overlay(
            dapi,
            af488,
            nuclei_labels,
            spot_labels,
            overlay_path,
            params,
        )
        overlay_paths.append(overlay_path)

        image_rows.append(
            image_summary_row(path, inspection, dapi, af488, nucleus_rows, spot_rows, nuclei_diag, spots_diag)
        )
        nucleus_rows_all.extend(nucleus_rows)
        spot_rows_all.extend(spot_rows)
        print(f"{path.name}: nuclei={len(nucleus_rows)} spots={len(spot_rows)}")

    write_csv(args.output_dir / "image_summary.csv", image_rows)
    write_csv(args.output_dir / "nucleus_measurements.csv", nucleus_rows_all)
    write_csv(args.output_dir / "spot_measurements.csv", spot_rows_all)
    make_qc_pdf(args.output_dir / "qc_overlay_contact_sheet.pdf", overlay_paths, args.qc_title)
    print(f"Wrote segmentation outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
