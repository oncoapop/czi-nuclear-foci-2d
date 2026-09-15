#!/usr/bin/env python3
"""Analyse two-channel TRF2/DAPI CZI files.

The red channel is quantified both as nuclear TRF2 intensity and as
puncta-like TRF2 objects. Puncta are not labelled as confirmed telomeres
without an independent telomere marker.
"""

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
import pandas as pd
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max


DEFAULT_ROOT = Path("/Volumes/Backup/TS Runs/TRF2Opt")
DEFAULT_OUTPUT = Path("output/trf2opt_trf2_dapi")


@dataclass(frozen=True)
class Parameters:
    trf2_channel_index: int = 0
    dapi_channel_index: int = 1
    nuclei_gaussian_sigma_px: float = 1.2
    nuclei_min_area_um2: float = 35.0
    nuclei_hole_area_um2: float = 20.0
    nuclei_watershed_min_distance_px: int = 10
    exclude_border_nuclei_from_summary: bool = True
    trf2_puncta_tophat_radius_px: int = 4
    trf2_puncta_min_area_px: int = 2
    trf2_puncta_threshold_mad_multiplier: float = 4.0
    trf2_min_intensity_above_local_background: float = 4.0
    trf2_positive_area_threshold_mad_multiplier: float = 3.0
    qc_display_percentile_low: float = 1.0
    qc_display_percentile_high: float = 99.8


def load_inspector():
    script = Path(__file__).with_name("inspect_czi.py")
    spec = importlib.util.spec_from_file_location("inspect_czi", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_sample(path: Path) -> dict[str, str]:
    match = re.match(
        r"^(?P<timepoint>\d+)\s*hr\s+20x\s+(?P<well>Well\d+)-(?P<replicate>\d+)\.czi$",
        path.name,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Filename does not match expected TRF2Opt pattern: {path.name}")
    timepoint = match.group("timepoint")
    well = match.group("well")
    replicate = match.group("replicate").zfill(2)
    sample_id = f"TRF2Opt_{timepoint}hr_{well}_rep{replicate}"
    return {
        "sample_id": sample_id,
        "file_name": path.name,
        "source_path": str(path),
        "timepoint_hr": timepoint,
        "condition": well,
        "well": well,
        "replicate": replicate,
    }


def selected_files(root: Path) -> list[Path]:
    paths = []
    for path in root.rglob("*.czi"):
        try:
            parse_sample(path)
        except ValueError:
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: (int(parse_sample(p)["timepoint_hr"]), parse_sample(p)["well"], p.name))


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


def read_channel_arrays(path: Path, inspection: dict, inspector) -> dict[int, np.ndarray]:
    arrays: dict[int, np.ndarray] = {}
    with path.open("rb") as handle:
        for index, block in enumerate(inspection["subblocks"]):
            if block["compression_code"] != 0:
                raise ValueError(f"Compressed CZI subblock is not supported: {path}")
            if block["dtype"] not in {"uint8", "uint16"}:
                raise ValueError(f"Expected Gray8/Gray16 data, found {block['dtype']}: {path}")
            dims = block["dimensions"]
            channel = int(dims.get("C", {}).get("start", index))
            width = int(dims["X"]["stored_size"])
            height = int(dims["Y"]["stored_size"])
            dtype = np.dtype(block["dtype"]).newbyteorder("<")
            handle.seek(block["data_offset"])
            raw = inspector.read_exact(handle, block["data_size"])
            arrays[channel] = np.frombuffer(raw, dtype=dtype).reshape((height, width))
    return arrays


def area_um2_to_px(area_um2: float, px_um_x: float, px_um_y: float) -> int:
    if not np.isfinite(px_um_x) or not np.isfinite(px_um_y) or px_um_x <= 0 or px_um_y <= 0:
        return max(1, int(round(area_um2)))
    return max(1, int(round(area_um2 / (px_um_x * px_um_y))))


def robust_threshold(values: np.ndarray, multiplier: float) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("inf")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    mad_sigma = 1.4826 * mad
    otsu = float(filters.threshold_otsu(values)) if np.unique(values).size > 1 else median
    return max(otsu, median + multiplier * mad_sigma)


def remove_objects_smaller_than(ar: np.ndarray, min_size: int) -> np.ndarray:
    """Remove objects with area smaller than min_size across scikit-image APIs."""
    try:
        return morphology.remove_small_objects(ar, max_size=max(0, int(min_size) - 1))
    except TypeError:
        return morphology.remove_small_objects(ar, min_size=min_size)


def remove_holes_smaller_than(ar: np.ndarray, min_size: int) -> np.ndarray:
    """Fill holes with area smaller than min_size across scikit-image APIs."""
    try:
        return morphology.remove_small_holes(ar, max_size=max(0, int(min_size) - 1))
    except TypeError:
        return morphology.remove_small_holes(ar, area_threshold=min_size)


def intensity_mean(region) -> float:
    if hasattr(region, "intensity_mean"):
        return float(region.intensity_mean)
    return float(region.mean_intensity)


def intensity_max(region) -> float:
    if hasattr(region, "intensity_max"):
        return float(region.intensity_max)
    return float(region.max_intensity)


def segment_nuclei(dapi: np.ndarray, px_um_x: float, px_um_y: float, params: Parameters) -> tuple[np.ndarray, dict]:
    smoothed = filters.gaussian(dapi, sigma=params.nuclei_gaussian_sigma_px, preserve_range=True)
    threshold = float(filters.threshold_otsu(smoothed)) if np.unique(smoothed).size > 1 else float(smoothed.mean())
    min_area = area_um2_to_px(params.nuclei_min_area_um2, px_um_x, px_um_y)
    hole_area = area_um2_to_px(params.nuclei_hole_area_um2, px_um_x, px_um_y)

    binary = smoothed > threshold
    binary = remove_objects_smaller_than(binary, min_area)
    binary = remove_holes_smaller_than(binary, hole_area)
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
    labels = segmentation.watershed(-distance, markers, mask=binary) if markers.max() else measure.label(binary)
    labels = remove_objects_smaller_than(labels, min_area)
    labels = measure.label(labels > 0).astype(np.uint16)
    return labels, {
        "dapi_otsu_threshold": threshold,
        "nuclei_min_area_px": int(min_area),
        "nuclei_hole_area_px": int(hole_area),
        "nuclei_watershed_markers": int(markers.max()),
        "nuclei_count": int(labels.max()),
    }


def segment_trf2_puncta(
    trf2: np.ndarray,
    nuclei_labels: np.ndarray,
    params: Parameters,
) -> tuple[np.ndarray, np.ndarray, dict]:
    nuclear_mask = nuclei_labels > 0
    footprint = morphology.disk(params.trf2_puncta_tophat_radius_px)
    enhanced = morphology.white_tophat(trf2, footprint=footprint)
    threshold = robust_threshold(enhanced[nuclear_mask], params.trf2_puncta_threshold_mad_multiplier)
    local_background = filters.gaussian(
        trf2,
        sigma=params.trf2_puncta_tophat_radius_px * 2,
        preserve_range=True,
    )
    above_background = trf2.astype(np.float32) - local_background.astype(np.float32)
    binary = (
        nuclear_mask
        & (enhanced >= threshold)
        & (above_background >= params.trf2_min_intensity_above_local_background)
    )
    binary = remove_objects_smaller_than(binary, params.trf2_puncta_min_area_px)
    labels = measure.label(binary).astype(np.uint16)
    return labels, enhanced, {
        "trf2_puncta_tophat_threshold": float(threshold),
        "trf2_puncta_min_area_px": int(params.trf2_puncta_min_area_px),
        "trf2_puncta_count": int(labels.max()),
    }


def display_u8(image: np.ndarray, params: Parameters) -> np.ndarray:
    low, high = np.percentile(image, [params.qc_display_percentile_low, params.qc_display_percentile_high])
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def touches_border(region, shape: tuple[int, int]) -> bool:
    min_row, min_col, max_row, max_col = region.bbox
    return min_row == 0 or min_col == 0 or max_row == shape[0] or max_col == shape[1]


def puncta_to_nucleus(puncta_labels: np.ndarray, nuclei_labels: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for region in measure.regionprops(puncta_labels):
        coords = region.coords
        nucleus_values = nuclei_labels[coords[:, 0], coords[:, 1]]
        nucleus_values = nucleus_values[nucleus_values > 0]
        if nucleus_values.size:
            mapping[int(region.label)] = int(np.bincount(nucleus_values.astype(np.int64)).argmax())
    return mapping


def measurement_rows(
    sample: dict,
    dapi: np.ndarray,
    trf2: np.ndarray,
    nuclei_labels: np.ndarray,
    puncta_labels: np.ndarray,
    enhanced: np.ndarray,
    px_um_x: float,
    px_um_y: float,
    params: Parameters,
) -> tuple[list[dict], list[dict]]:
    area_scale = px_um_x * px_um_y if np.isfinite(px_um_x * px_um_y) else float("nan")
    nuclear_mask = nuclei_labels > 0
    non_nuclear = trf2[~nuclear_mask]
    trf2_background_median = float(np.median(non_nuclear)) if non_nuclear.size else float(np.median(trf2))
    trf2_positive_threshold = robust_threshold(
        trf2[nuclear_mask],
        params.trf2_positive_area_threshold_mad_multiplier,
    )

    by_nucleus: dict[int, list] = {}
    focus_to_nucleus = puncta_to_nucleus(puncta_labels, nuclei_labels)
    puncta_rows = []
    for punctum in measure.regionprops(puncta_labels, intensity_image=trf2):
        nucleus_label = focus_to_nucleus.get(int(punctum.label), 0)
        by_nucleus.setdefault(nucleus_label, []).append(punctum)
        puncta_rows.append(
            {
                **sample,
                "nucleus_label": nucleus_label,
                "trf2_punctum_label": int(punctum.label),
                "trf2_punctum_area_px": int(punctum.area),
                "trf2_punctum_area_um2": float(punctum.area * area_scale),
                "trf2_punctum_mean_intensity": intensity_mean(punctum),
                "trf2_punctum_max_intensity": intensity_max(punctum),
                "trf2_punctum_centroid_y_px": float(punctum.centroid[0]),
                "trf2_punctum_centroid_x_px": float(punctum.centroid[1]),
            }
        )

    nucleus_rows = []
    for nucleus in measure.regionprops(nuclei_labels, intensity_image=dapi):
        label = int(nucleus.label)
        mask = nuclei_labels == label
        trf2_values = trf2[mask].astype(np.float32)
        dapi_values = dapi[mask].astype(np.float32)
        enhanced_values = enhanced[mask].astype(np.float32)
        puncta = by_nucleus.get(label, [])
        border = touches_border(nucleus, nuclei_labels.shape)
        valid = not border if params.exclude_border_nuclei_from_summary else True
        positive_area_px = int((trf2_values >= trf2_positive_threshold).sum())
        nucleus_rows.append(
            {
                **sample,
                "nucleus_label": label,
                "valid_for_summary": "TRUE" if valid else "FALSE",
                "touches_image_border": "TRUE" if border else "FALSE",
                "nucleus_area_px": int(nucleus.area),
                "nucleus_area_um2": float(nucleus.area * area_scale),
                "nucleus_centroid_y_px": float(nucleus.centroid[0]),
                "nucleus_centroid_x_px": float(nucleus.centroid[1]),
                "nucleus_mean_dapi_intensity": float(dapi_values.mean()),
                "nucleus_mean_trf2_intensity": float(trf2_values.mean()),
                "nucleus_median_trf2_intensity": float(np.median(trf2_values)),
                "nucleus_integrated_trf2_intensity": float(trf2_values.sum()),
                "nucleus_background_subtracted_mean_trf2": float(trf2_values.mean() - trf2_background_median),
                "nucleus_background_subtracted_integrated_trf2": float(
                    (trf2_values - trf2_background_median).sum()
                ),
                "nucleus_mean_trf2_tophat_intensity": float(enhanced_values.mean()),
                "trf2_positive_threshold": float(trf2_positive_threshold),
                "trf2_positive_area_px": positive_area_px,
                "trf2_positive_area_fraction": float(positive_area_px / nucleus.area) if nucleus.area else 0.0,
                "trf2_puncta_count": len(puncta),
                "trf2_puncta_total_area_px": int(sum(p.area for p in puncta)),
                "trf2_puncta_total_area_um2": float(sum(p.area for p in puncta) * area_scale),
                "trf2_puncta_mean_intensity_mean": float(np.mean([intensity_mean(p) for p in puncta]))
                if puncta
                else 0.0,
                "image_trf2_background_median": trf2_background_median,
            }
        )
    return nucleus_rows, puncta_rows


def image_summary_row(
    sample: dict,
    inspection: dict,
    dapi: np.ndarray,
    trf2: np.ndarray,
    nucleus_rows: list[dict],
    puncta_rows: list[dict],
    nuclei_diag: dict,
    puncta_diag: dict,
    params: Parameters,
) -> dict:
    valid_rows = [row for row in nucleus_rows if row["valid_for_summary"] == "TRUE"]
    rows_for_means = valid_rows if params.exclude_border_nuclei_from_summary else nucleus_rows
    return {
        **sample,
        "acquisition_datetime": inspection["metadata"]["image"].get("AcquisitionDateAndTime", ""),
        "dapi_channel_metadata": channel_metadata(inspection, params.dapi_channel_index),
        "trf2_channel_metadata": channel_metadata(inspection, params.trf2_channel_index),
        "nuclei_count_total": len(nucleus_rows),
        "nuclei_count_valid_for_summary": len(rows_for_means),
        "nuclei_excluded_border_count": len(nucleus_rows) - len(valid_rows),
        "trf2_puncta_count_total": len(puncta_rows),
        "mean_trf2_puncta_per_valid_nucleus": float(np.mean([r["trf2_puncta_count"] for r in rows_for_means]))
        if rows_for_means
        else 0.0,
        "mean_nuclear_trf2_intensity_valid_nuclei": float(
            np.mean([r["nucleus_mean_trf2_intensity"] for r in rows_for_means])
        )
        if rows_for_means
        else 0.0,
        "mean_background_subtracted_trf2_valid_nuclei": float(
            np.mean([r["nucleus_background_subtracted_mean_trf2"] for r in rows_for_means])
        )
        if rows_for_means
        else 0.0,
        "mean_trf2_positive_area_fraction_valid_nuclei": float(
            np.mean([r["trf2_positive_area_fraction"] for r in rows_for_means])
        )
        if rows_for_means
        else 0.0,
        "raw_dapi_mean_intensity": float(dapi.mean()),
        "raw_dapi_max_intensity": float(dapi.max()),
        "raw_trf2_mean_intensity": float(trf2.mean()),
        "raw_trf2_max_intensity": float(trf2.max()),
        "dapi_saturated_pixel_fraction": float((dapi == np.iinfo(dapi.dtype).max).mean())
        if np.issubdtype(dapi.dtype, np.integer)
        else float("nan"),
        "trf2_saturated_pixel_fraction": float((trf2 == np.iinfo(trf2.dtype).max).mean())
        if np.issubdtype(trf2.dtype, np.integer)
        else float("nan"),
        **nuclei_diag,
        **puncta_diag,
    }


def source_metadata_row(path: Path, inspection: dict, params: Parameters) -> dict:
    image = inspection["metadata"].get("image") or {}
    px_um_x, px_um_y = pixel_size_um(inspection["metadata"])
    sample = parse_sample(path)
    return {
        **sample,
        "source_size_bytes": inspection.get("source_size_bytes"),
        "source_sha256": inspection.get("source_sha256"),
        "size_x_px": image.get("SizeX", ""),
        "size_y_px": image.get("SizeY", ""),
        "size_z": image.get("SizeZ", ""),
        "size_c": image.get("SizeC", ""),
        "pixel_type": image.get("PixelType", ""),
        "component_bit_count": image.get("ComponentBitCount", ""),
        "pixel_size_x_um": px_um_x,
        "pixel_size_y_um": px_um_y,
        "trf2_channel_metadata": channel_metadata(inspection, params.trf2_channel_index),
        "dapi_channel_metadata": channel_metadata(inspection, params.dapi_channel_index),
        "compression": ";".join(sorted({str(block.get("compression", "")) for block in inspection["subblocks"]})),
        "subblock_count": len(inspection["subblocks"]),
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


def save_label_tiff(labels: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    Image.fromarray(labels).save(path, format="TIFF")


def save_qc_overlay(
    dapi: np.ndarray,
    trf2: np.ndarray,
    nuclei_labels: np.ndarray,
    puncta_labels: np.ndarray,
    path: Path,
    params: Parameters,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    rgb = np.zeros((*dapi.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = display_u8(trf2, params)
    rgb[:, :, 2] = display_u8(dapi, params)
    nuclei_boundary = segmentation.find_boundaries(nuclei_labels, mode="outer")
    puncta_boundary = segmentation.find_boundaries(puncta_labels, mode="outer")
    rgb[nuclei_boundary] = np.array([255, 255, 0], dtype=np.uint8)
    rgb[puncta_boundary] = np.array([0, 255, 0], dtype=np.uint8)

    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 5, 520, 58), fill=(0, 0, 0))
    draw.text((12, 12), "Blue=DAPI, red=TRF2/Rhodamine Red-X", fill=(255, 255, 255))
    draw.text((12, 32), "Yellow=nucleus boundary, green=TRF2 puncta-like boundary", fill=(255, 255, 255))
    image.save(path)


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
            "Blue=DAPI; red=TRF2; yellow=nucleus boundary; green=TRF2 puncta-like boundary.",
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


def write_method_notes(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(
        "\n".join(
            [
                "# TRF2/DAPI analysis notes",
                "",
                "Verified channel assignment from CZI metadata:",
                "- C0: Rhodamine Red-X, used as TRF2.",
                "- C1: DAPI, used for nuclear segmentation.",
                "",
                "Interpretation boundary:",
                "- TRF2 is a telomere-binding shelterin protein and a working antibody can appear punctate in nuclei.",
                "- This dataset has TRF2 plus DAPI only. Puncta-like red objects are therefore quantified as TRF2 puncta-like objects, not as independently confirmed telomere ends.",
                "- Nuclear red intensity, background-subtracted intensity, positive red area fraction, and puncta-like counts are all reported so poor or diffuse optimisation conditions remain interpretable.",
                "",
                "Summary statistics exclude nuclei touching the image border by default, while per-nucleus rows retain those nuclei with flags.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_condition_summary(path: Path, nucleus_rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if not nucleus_rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    nuclei = pd.DataFrame(nucleus_rows)
    valid = nuclei[nuclei["valid_for_summary"] == "TRUE"].copy()
    if valid.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    metrics = [
        "nucleus_mean_trf2_intensity",
        "nucleus_background_subtracted_mean_trf2",
        "nucleus_integrated_trf2_intensity",
        "trf2_positive_area_fraction",
        "trf2_puncta_count",
        "trf2_puncta_total_area_px",
    ]
    grouped = valid.groupby(["timepoint_hr", "condition", "well"], sort=True)
    summary = grouped.agg(
        n_valid_nuclei=("nucleus_label", "count"),
        **{f"mean_{metric}": (metric, "mean") for metric in metrics},
        **{f"median_{metric}": (metric, "median") for metric in metrics},
    ).reset_index()
    summary.to_csv(path, index=False)


def analyse(root: Path, output_dir: Path, params: Parameters) -> None:
    files = selected_files(root)
    if not files:
        raise ValueError(f"No matching CZI files found under {root}")

    inspector = load_inspector()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parameters.json").write_text(
        json.dumps(asdict(params), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_method_notes(output_dir / "method_notes.md")

    image_rows: list[dict] = []
    nucleus_rows_all: list[dict] = []
    puncta_rows_all: list[dict] = []
    metadata_rows: list[dict] = []
    overlay_paths: list[Path] = []

    for path in files:
        sample = parse_sample(path)
        inspection = inspector.inspect(path, include_stats=False, preview_dir=None, plane_dir=None)
        arrays = read_channel_arrays(path, inspection, inspector)
        trf2 = arrays[params.trf2_channel_index]
        dapi = arrays[params.dapi_channel_index]
        px_um_x, px_um_y = pixel_size_um(inspection["metadata"])

        nuclei_labels, nuclei_diag = segment_nuclei(dapi, px_um_x, px_um_y, params)
        puncta_labels, enhanced, puncta_diag = segment_trf2_puncta(trf2, nuclei_labels, params)
        nucleus_rows, puncta_rows = measurement_rows(
            sample,
            dapi,
            trf2,
            nuclei_labels,
            puncta_labels,
            enhanced,
            px_um_x,
            px_um_y,
            params,
        )

        sid = sample["sample_id"]
        save_label_tiff(nuclei_labels, output_dir / "masks" / f"{sid}_nuclei_labels.tif")
        save_label_tiff(puncta_labels, output_dir / "masks" / f"{sid}_trf2_puncta_labels.tif")
        overlay_path = output_dir / "qc_overlays" / f"{sid}_qc_overlay.png"
        save_qc_overlay(dapi, trf2, nuclei_labels, puncta_labels, overlay_path, params)
        overlay_paths.append(overlay_path)

        image_rows.append(
            image_summary_row(
                sample,
                inspection,
                dapi,
                trf2,
                nucleus_rows,
                puncta_rows,
                nuclei_diag,
                puncta_diag,
                params,
            )
        )
        nucleus_rows_all.extend(nucleus_rows)
        puncta_rows_all.extend(puncta_rows)
        metadata_rows.append(source_metadata_row(path, inspection, params))
        valid_nuclei = sum(1 for row in nucleus_rows if row["valid_for_summary"] == "TRUE")
        print(f"{path.name}: nuclei={len(nucleus_rows)} valid={valid_nuclei} trf2_puncta={len(puncta_rows)}")

    write_csv(output_dir / "source_metadata.csv", metadata_rows)
    write_csv(output_dir / "image_summary.csv", image_rows)
    write_csv(output_dir / "nucleus_measurements.csv", nucleus_rows_all)
    write_csv(output_dir / "trf2_puncta_measurements.csv", puncta_rows_all)
    write_condition_summary(output_dir / "condition_summary.csv", nucleus_rows_all)
    make_qc_pdf(output_dir / "qc_overlay_contact_sheet.pdf", overlay_paths, "TRF2Opt TRF2/DAPI QC")
    print(f"Wrote TRF2/DAPI analysis outputs to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            parser.error(f"Refusing to overwrite existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    try:
        analyse(args.root, output_dir, Parameters())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
