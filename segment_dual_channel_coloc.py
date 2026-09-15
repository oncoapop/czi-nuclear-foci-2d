#!/usr/bin/env python3
"""Segment AF488 and RhReX nuclear foci and call co-localised spots."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


@dataclass(frozen=True)
class DualParameters:
    dapi_channel_index: int = 2
    af488_channel_index: int = 1
    rhrex_channel_index: int = 0
    af488_tophat_radius_px: int = 4
    af488_min_area_px: int = 3
    af488_threshold_mad_multiplier: float = 4.0
    af488_min_intensity_above_local_background: float = 8.0
    rhrex_tophat_radius_px: int = 8
    rhrex_min_area_px: int = 8
    rhrex_threshold_mad_multiplier: float = 4.0
    rhrex_min_intensity_above_local_background: float = 8.0
    colocalization_dilation_px: int = 1
    qc_display_percentile_low: float = 1.0
    qc_display_percentile_high: float = 99.8


CONTROL_CONDITIONS = {"Aqueous", "DMSO", "NAH2PO4"}
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


def load_module(name: str, file_name: str):
    script = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def robust_threshold(values: np.ndarray, multiplier: float) -> float:
    if values.size == 0:
        return float("inf")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    mad_sigma = 1.4826 * mad
    otsu = float(filters.threshold_otsu(values)) if np.unique(values).size > 1 else median
    return max(otsu, median + multiplier * mad_sigma)


def segment_spots(
    image: np.ndarray,
    nuclei_labels: np.ndarray,
    radius_px: int,
    min_area_px: int,
    threshold_mad_multiplier: float,
    min_intensity_above_background: float,
) -> tuple[np.ndarray, dict]:
    nuclear_mask = nuclei_labels > 0
    footprint = morphology.disk(radius_px)
    enhanced = morphology.white_tophat(image, footprint=footprint)
    threshold = robust_threshold(enhanced[nuclear_mask], threshold_mad_multiplier)
    local_background = filters.gaussian(image, sigma=radius_px * 2, preserve_range=True)
    above_background = image.astype(np.float32) - local_background.astype(np.float32)
    binary = (
        nuclear_mask
        & (enhanced >= threshold)
        & (above_background >= min_intensity_above_background)
    )
    binary = morphology.remove_small_objects(binary, min_size=min_area_px)
    labels = measure.label(binary).astype(np.uint16)
    return labels, {
        "threshold": float(threshold),
        "min_area_px": int(min_area_px),
        "tophat_radius_px": int(radius_px),
        "spot_count": int(labels.max()),
    }


def display_u8(image: np.ndarray, params: DualParameters) -> np.ndarray:
    low, high = np.percentile(image, [params.qc_display_percentile_low, params.qc_display_percentile_high])
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)
    return np.clip((image.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def pixel_size_um(metadata: dict) -> tuple[float, float]:
    values: dict[str, float] = {}
    for item in metadata.get("scaling") or []:
        if item.get("Id") in {"X", "Y"} and item.get("Value"):
            values[item["Id"]] = float(item["Value"]) * 1_000_000.0
    return values.get("X", float("nan")), values.get("Y", float("nan"))


def area_scale(px_um_x: float, px_um_y: float) -> float:
    scale = px_um_x * px_um_y
    return scale if np.isfinite(scale) else float("nan")


def spot_to_nucleus_labels(spot_labels: np.ndarray, nuclei_labels: np.ndarray) -> dict[int, int]:
    mapping = {}
    for region in measure.regionprops(spot_labels):
        coords = region.coords
        values = nuclei_labels[coords[:, 0], coords[:, 1]]
        values = values[values > 0]
        mapping[int(region.label)] = int(np.bincount(values.astype(np.int64)).argmax()) if values.size else 0
    return mapping


def colocalization_calls(
    af488_labels: np.ndarray,
    rhrex_labels: np.ndarray,
    nuclei_labels: np.ndarray,
    dilation_px: int,
) -> tuple[dict[int, set[int]], set[int], set[int], np.ndarray]:
    structure = morphology.disk(dilation_px)
    af_binary_dilated = morphology.binary_dilation(af488_labels > 0, footprint=structure)
    rh_binary_dilated = morphology.binary_dilation(rhrex_labels > 0, footprint=structure)
    rh_to_af: dict[int, set[int]] = {}
    coloc_rh = set()
    coloc_af = set()

    for rh_region in measure.regionprops(rhrex_labels):
        coords = rh_region.coords
        if not af_binary_dilated[coords[:, 0], coords[:, 1]].any():
            continue
        rh_label = int(rh_region.label)
        nearby_mask = np.zeros(rhrex_labels.shape, dtype=bool)
        nearby_mask[coords[:, 0], coords[:, 1]] = True
        nearby_mask = morphology.binary_dilation(nearby_mask, footprint=structure)
        af_labels = set(int(v) for v in np.unique(af488_labels[nearby_mask]) if v > 0)
        if af_labels:
            rh_to_af[rh_label] = af_labels
            coloc_rh.add(rh_label)
            coloc_af.update(af_labels)

    coloc_rh_labels = np.where(np.isin(rhrex_labels, list(coloc_rh)), rhrex_labels, 0).astype(np.uint16)
    return rh_to_af, coloc_rh, coloc_af, coloc_rh_labels


def save_label_tiff(labels: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    Image.fromarray(labels).save(path, format="TIFF")


def save_qc_overlay(
    dapi: np.ndarray,
    af488: np.ndarray,
    rhrex: np.ndarray,
    nuclei_labels: np.ndarray,
    af488_labels: np.ndarray,
    rhrex_labels: np.ndarray,
    coloc_rh_labels: np.ndarray,
    path: Path,
    params: DualParameters,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    rgb = np.zeros((*dapi.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = display_u8(rhrex, params)
    rgb[:, :, 1] = display_u8(af488, params)
    rgb[:, :, 2] = display_u8(dapi, params)

    nuclei_boundary = segmentation.find_boundaries(nuclei_labels, mode="outer")
    af_boundary = segmentation.find_boundaries(af488_labels, mode="outer")
    rh_boundary = segmentation.find_boundaries(rhrex_labels, mode="outer")
    coloc_boundary = segmentation.find_boundaries(coloc_rh_labels, mode="outer")
    rgb[nuclei_boundary] = np.array([255, 255, 0], dtype=np.uint8)
    rgb[af_boundary] = np.array([0, 255, 255], dtype=np.uint8)
    rgb[rh_boundary] = np.array([255, 128, 0], dtype=np.uint8)
    rgb[coloc_boundary] = np.array([255, 0, 255], dtype=np.uint8)

    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 5, 560, 74), fill=(0, 0, 0))
    draw.text((12, 12), "Blue=DAPI/T3, Green=AF488/T2, Red=RhReX/T1 putative gH2AX", fill=(255, 255, 255))
    draw.text((12, 31), "Yellow=nucleus, cyan=AF488 spot, orange=RhReX focus", fill=(255, 255, 255))
    draw.text((12, 50), "Magenta=RhReX focus overlapping AF488 spot after 1 px dilation", fill=(255, 255, 255))
    image.save(path)


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


def measure_nuclei_and_spots(
    path: Path,
    inspection: dict,
    dapi: np.ndarray,
    af488: np.ndarray,
    rhrex: np.ndarray,
    nuclei_labels: np.ndarray,
    af488_labels: np.ndarray,
    rhrex_labels: np.ndarray,
    rh_to_af: dict[int, set[int]],
    coloc_rh: set[int],
    coloc_af: set[int],
    base,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    condition, timepoint, replicate = base.parse_condition(path)
    sid = base.sample_id(path)
    px_um_x, px_um_y = pixel_size_um(inspection["metadata"])
    scale = area_scale(px_um_x, px_um_y)
    af_to_nucleus = spot_to_nucleus_labels(af488_labels, nuclei_labels)
    rh_to_nucleus = spot_to_nucleus_labels(rhrex_labels, nuclei_labels)

    af_by_nucleus: dict[int, list] = {}
    for label, nucleus in af_to_nucleus.items():
        af_by_nucleus.setdefault(nucleus, []).append(label)
    rh_by_nucleus: dict[int, list] = {}
    coloc_rh_by_nucleus: dict[int, list] = {}
    for label, nucleus in rh_to_nucleus.items():
        rh_by_nucleus.setdefault(nucleus, []).append(label)
        if label in coloc_rh:
            coloc_rh_by_nucleus.setdefault(nucleus, []).append(label)

    nucleus_rows = []
    for nucleus in measure.regionprops(nuclei_labels, intensity_image=dapi):
        label = int(nucleus.label)
        mask = nuclei_labels == label
        af_count = len(af_by_nucleus.get(label, []))
        rh_count = len(rh_by_nucleus.get(label, []))
        coloc_count = len(coloc_rh_by_nucleus.get(label, []))
        nucleus_rows.append(
            {
                "sample_id": sid,
                "file_name": path.name,
                "source_path": str(path),
                "acquisition_date_folder": base.acquisition_date_folder(path),
                "condition": condition,
                "timepoint_hr": timepoint,
                "replicate": replicate,
                "nucleus_label": label,
                "nucleus_area_px": int(nucleus.area),
                "nucleus_area_um2": float(nucleus.area * scale),
                "nucleus_mean_dapi_intensity": float(nucleus.mean_intensity),
                "nucleus_mean_af488_intensity": float(af488[mask].mean()),
                "nucleus_mean_rhrex_intensity": float(rhrex[mask].mean()),
                "af488_spots_count": af_count,
                "rhrex_spots_count": rh_count,
                "colocalized_rhrex_spots_count": coloc_count,
                "af488_has_colocalized_spots_count": sum(1 for af in af_by_nucleus.get(label, []) if af in coloc_af),
                "nucleus_centroid_y_px": float(nucleus.centroid[0]),
                "nucleus_centroid_x_px": float(nucleus.centroid[1]),
            }
        )

    af_rows = []
    for spot in measure.regionprops(af488_labels, intensity_image=af488):
        label = int(spot.label)
        nucleus = af_to_nucleus.get(label, 0)
        af_rows.append(
            {
                "sample_id": sid,
                "file_name": path.name,
                "condition": condition,
                "timepoint_hr": timepoint,
                "replicate": replicate,
                "spot_channel": "AF488-T2",
                "spot_label": label,
                "nucleus_label": nucleus,
                "is_colocalized_with_rhrex": "TRUE" if label in coloc_af else "FALSE",
                "spot_area_px": int(spot.area),
                "spot_area_um2": float(spot.area * scale),
                "spot_mean_intensity": float(spot.mean_intensity),
                "spot_max_intensity": float(spot.max_intensity),
                "spot_centroid_y_px": float(spot.centroid[0]),
                "spot_centroid_x_px": float(spot.centroid[1]),
            }
        )

    rh_rows = []
    for spot in measure.regionprops(rhrex_labels, intensity_image=rhrex):
        label = int(spot.label)
        nucleus = rh_to_nucleus.get(label, 0)
        rh_rows.append(
            {
                "sample_id": sid,
                "file_name": path.name,
                "condition": condition,
                "timepoint_hr": timepoint,
                "replicate": replicate,
                "spot_channel": "RhReX-T1_putative_gH2AX",
                "spot_label": label,
                "nucleus_label": nucleus,
                "is_colocalized_with_af488": "TRUE" if label in coloc_rh else "FALSE",
                "overlapping_af488_labels": ";".join(str(v) for v in sorted(rh_to_af.get(label, []))),
                "spot_area_px": int(spot.area),
                "spot_area_um2": float(spot.area * scale),
                "spot_mean_intensity": float(spot.mean_intensity),
                "spot_max_intensity": float(spot.max_intensity),
                "spot_centroid_y_px": float(spot.centroid[0]),
                "spot_centroid_x_px": float(spot.centroid[1]),
            }
        )

    pair_rows = []
    for rh_label, af_labels in rh_to_af.items():
        for af_label in sorted(af_labels):
            pair_rows.append(
                {
                    "sample_id": sid,
                    "file_name": path.name,
                    "condition": condition,
                    "timepoint_hr": timepoint,
                    "replicate": replicate,
                    "nucleus_label": rh_to_nucleus.get(rh_label, 0),
                    "rhrex_spot_label": rh_label,
                    "af488_spot_label": af_label,
                }
            )
    return nucleus_rows, af_rows, rh_rows, pair_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def image_summary_row(
    path: Path,
    inspection: dict,
    nucleus_rows: list[dict],
    af_rows: list[dict],
    rh_rows: list[dict],
    pair_rows: list[dict],
    af_diag: dict,
    rh_diag: dict,
    base,
    params: DualParameters,
) -> dict:
    condition, timepoint, replicate = base.parse_condition(path)
    nuclei = len(nucleus_rows)
    return {
        "sample_id": base.sample_id(path),
        "file_name": path.name,
        "source_path": str(path),
        "acquisition_date_folder": base.acquisition_date_folder(path),
        "condition": condition,
        "timepoint_hr": timepoint,
        "replicate": replicate,
        "acquisition_datetime": inspection["metadata"]["image"].get("AcquisitionDateAndTime", ""),
        "channel_0_metadata": channel_metadata(inspection, params.rhrex_channel_index),
        "channel_1_metadata": channel_metadata(inspection, params.af488_channel_index),
        "channel_2_metadata": channel_metadata(inspection, params.dapi_channel_index),
        "nuclei_count": nuclei,
        "af488_spots_count": len(af_rows),
        "rhrex_spots_count": len(rh_rows),
        "colocalized_rhrex_spots_count": len({row["rhrex_spot_label"] for row in pair_rows}),
        "colocalized_pairs_count": len(pair_rows),
        "mean_af488_spots_per_nucleus": float(np.mean([r["af488_spots_count"] for r in nucleus_rows])) if nucleus_rows else 0.0,
        "mean_rhrex_spots_per_nucleus": float(np.mean([r["rhrex_spots_count"] for r in nucleus_rows])) if nucleus_rows else 0.0,
        "mean_colocalized_rhrex_spots_per_nucleus": float(np.mean([r["colocalized_rhrex_spots_count"] for r in nucleus_rows])) if nucleus_rows else 0.0,
        "af488_threshold": af_diag["threshold"],
        "rhrex_threshold": rh_diag["threshold"],
    }


def make_qc_pdf(path: Path, overlay_paths: list[Path], title: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    width, height = landscape(A4)
    pdf = canvas.Canvas(str(path), pagesize=landscape(A4))
    thumb_w, thumb_h = 180, 180
    per_page = 8
    for page_start in range(0, len(overlay_paths), per_page):
        page_paths = overlay_paths[page_start : page_start + per_page]
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(36, height - 32, title)
        pdf.setFont("Helvetica", 8)
        pdf.drawString(36, height - 46, "RGB: RhReX/AF488/DAPI. Magenta boundaries mark co-localised RhReX foci.")
        pdf.drawRightString(width - 36, height - 46, f"Page {page_start // per_page + 1}")
        for index, overlay in enumerate(page_paths):
            col = index % 4
            row = index // 4
            x = 36 + col * 198
            y = height - 70 - row * 216 - thumb_h
            pdf.drawImage(ImageReader(str(overlay)), x, y, thumb_w, thumb_h, preserveAspectRatio=True, anchor="c")
            pdf.drawString(x, y - 11, overlay.stem.replace("_dual_qc_overlay", ""))
        pdf.showPage()
    pdf.save()


def threshold_table(df: pd.DataFrame, count_column: str, output: Path) -> pd.DataFrame:
    rows = []
    treated = ~df["condition"].isin(CONTROL_CONDITIONS)
    controls = df["condition"].isin(CONTROL_CONDITIONS)
    max_count = int(df[count_column].max()) if len(df) else 0
    for threshold in range(0, max_count + 2):
        positive = df[count_column] >= threshold
        tp = int((positive & treated).sum())
        fp = int((positive & controls).sum())
        tn = int((~positive & controls).sum())
        fn = int((~positive & treated).sum())
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        rows.append(
            {
                "metric": count_column,
                "positive_if_count_ge": threshold,
                "true_positive_treated_nuclei": tp,
                "false_positive_control_nuclei": fp,
                "true_negative_control_nuclei": tn,
                "false_negative_treated_nuclei": fn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "youden_j": sensitivity + specificity - 1.0,
                "control_positive_fraction": fp / int(controls.sum()) if controls.sum() else 0.0,
                "treated_positive_fraction": tp / int(treated.sum()) if treated.sum() else 0.0,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output, index=False)
    return table


def condition_summary(df: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    for (timepoint, condition), group in df.groupby(["timepoint_hr", "condition"], observed=True):
        rows.append(
            {
                "timepoint_hr": timepoint,
                "condition": condition,
                "n_nuclei": int(len(group)),
                "mean_af488_spots_per_nucleus": float(group["af488_spots_count"].mean()),
                "mean_rhrex_spots_per_nucleus": float(group["rhrex_spots_count"].mean()),
                "mean_colocalized_rhrex_spots_per_nucleus": float(group["colocalized_rhrex_spots_count"].mean()),
                "median_colocalized_rhrex_spots_per_nucleus": float(group["colocalized_rhrex_spots_count"].median()),
                "fraction_nuclei_ge1_colocalized": float((group["colocalized_rhrex_spots_count"] >= 1).mean()),
                "fraction_nuclei_ge2_colocalized": float((group["colocalized_rhrex_spots_count"] >= 2).mean()),
                "fraction_nuclei_ge3_colocalized": float((group["colocalized_rhrex_spots_count"] >= 3).mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output, index=False)
    return summary


def plot_metric(df: pd.DataFrame, metric: str, threshold: int, title: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    timepoints = sorted(df["timepoint_hr"].astype(str).unique(), key=lambda x: int(x))
    colours = dict(zip(CONDITION_ORDER, plt.cm.tab10(np.linspace(0, 1, len(CONDITION_ORDER)))))
    fig, axes = plt.subplots(1, len(timepoints), figsize=(16, 7), sharey=True)
    if len(timepoints) == 1:
        axes = [axes]
    rng = np.random.default_rng(12345)
    for ax, timepoint in zip(axes, timepoints):
        sub = df[df["timepoint_hr"].astype(str) == timepoint]
        for idx, condition in enumerate(CONDITION_ORDER):
            values = sub.loc[sub["condition"] == condition, metric].to_numpy(dtype=float)
            if values.size == 0:
                continue
            x = np.full(values.shape, idx, dtype=float) + rng.uniform(-0.28, 0.28, size=values.shape)
            ax.scatter(x, values, s=10, alpha=0.45, color=colours[condition], edgecolors="none")
            ax.plot([idx - 0.25, idx + 0.25], [np.median(values), np.median(values)], color="black", linewidth=2)
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2)
        ax.set_title(f"{timepoint} hr")
        ax.set_xticks(range(len(CONDITION_ORDER)))
        ax.set_xticklabels(CONDITION_ORDER, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(metric.replace("_", " "))
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qc-title", default="Dual channel co-localisation QC")
    parser.add_argument("--plot-title", default="Dual channel nuclear spot counts")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        if not args.overwrite:
            parser.error(f"Refusing to overwrite existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    params = DualParameters()
    (args.output_dir / "parameters.json").write_text(json.dumps(asdict(params), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    base = load_module("segment_nuclei_spots_base", "segment_nuclei_spots.py")
    inspector = base.load_inspector()
    files = base.selected_files_from_root(args.root)
    if args.limit:
        files = files[: args.limit]

    image_rows, nucleus_rows_all, af_rows_all, rh_rows_all, pair_rows_all = [], [], [], [], []
    overlay_paths = []
    for path in files:
        inspection = inspector.inspect(path, True, None)
        arrays = base.read_channel_arrays(path, inspection, inspector)
        dapi = arrays[params.dapi_channel_index]
        af488 = arrays[params.af488_channel_index]
        rhrex = arrays[params.rhrex_channel_index]
        px_um_x, px_um_y = pixel_size_um(inspection["metadata"])
        nuclei_params = base.Parameters()
        nuclei_labels, _ = base.segment_nuclei(dapi, px_um_x, px_um_y, nuclei_params)
        af_labels, af_diag = segment_spots(
            af488,
            nuclei_labels,
            params.af488_tophat_radius_px,
            params.af488_min_area_px,
            params.af488_threshold_mad_multiplier,
            params.af488_min_intensity_above_local_background,
        )
        rh_labels, rh_diag = segment_spots(
            rhrex,
            nuclei_labels,
            params.rhrex_tophat_radius_px,
            params.rhrex_min_area_px,
            params.rhrex_threshold_mad_multiplier,
            params.rhrex_min_intensity_above_local_background,
        )
        rh_to_af, coloc_rh, coloc_af, coloc_rh_labels = colocalization_calls(
            af_labels,
            rh_labels,
            nuclei_labels,
            params.colocalization_dilation_px,
        )
        nucleus_rows, af_rows, rh_rows, pair_rows = measure_nuclei_and_spots(
            path,
            inspection,
            dapi,
            af488,
            rhrex,
            nuclei_labels,
            af_labels,
            rh_labels,
            rh_to_af,
            coloc_rh,
            coloc_af,
            base,
        )
        sid = base.sample_id(path)
        save_label_tiff(nuclei_labels, args.output_dir / "masks" / f"{sid}_nuclei_labels.tif")
        save_label_tiff(af_labels, args.output_dir / "masks" / f"{sid}_af488_spot_labels.tif")
        save_label_tiff(rh_labels, args.output_dir / "masks" / f"{sid}_rhrex_putative_gH2AX_labels.tif")
        save_label_tiff(coloc_rh_labels, args.output_dir / "masks" / f"{sid}_colocalized_rhrex_labels.tif")
        overlay_path = args.output_dir / "qc_overlays" / f"{sid}_dual_qc_overlay.png"
        save_qc_overlay(dapi, af488, rhrex, nuclei_labels, af_labels, rh_labels, coloc_rh_labels, overlay_path, params)
        overlay_paths.append(overlay_path)

        image_rows.append(image_summary_row(path, inspection, nucleus_rows, af_rows, rh_rows, pair_rows, af_diag, rh_diag, base, params))
        nucleus_rows_all.extend(nucleus_rows)
        af_rows_all.extend(af_rows)
        rh_rows_all.extend(rh_rows)
        pair_rows_all.extend(pair_rows)
        print(f"{path.name}: nuclei={len(nucleus_rows)} AF488={len(af_rows)} RhReX={len(rh_rows)} coloc_RhReX={len(set(r['rhrex_spot_label'] for r in pair_rows))}")

    write_csv(args.output_dir / "image_summary_dual_channel.csv", image_rows)
    write_csv(args.output_dir / "nucleus_dual_channel_measurements.csv", nucleus_rows_all)
    write_csv(args.output_dir / "af488_spot_measurements.csv", af_rows_all)
    write_csv(args.output_dir / "rhrex_putative_gH2AX_spot_measurements.csv", rh_rows_all)
    write_csv(args.output_dir / "colocalized_spot_pairs.csv", pair_rows_all)
    make_qc_pdf(args.output_dir / "dual_channel_qc_contact_sheet.pdf", overlay_paths, args.qc_title)

    df = pd.DataFrame(nucleus_rows_all)
    plot_dir = args.output_dir / "pooled_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    condition_summary(df, plot_dir / "condition_timepoint_dual_channel_summary.csv")
    thresholds = {}
    for metric in ["rhrex_spots_count", "colocalized_rhrex_spots_count"]:
        table = threshold_table(df, metric, plot_dir / f"threshold_performance_{metric}.csv")
        for timepoint in sorted(df["timepoint_hr"].astype(str).unique(), key=lambda x: int(x)):
            threshold_table(
                df[df["timepoint_hr"].astype(str) == timepoint],
                metric,
                plot_dir / f"threshold_performance_{metric}_{timepoint}hr.csv",
            )
        best = table.sort_values(["youden_j", "specificity", "sensitivity"], ascending=False).iloc[0]
        thresholds[metric] = int(best["positive_if_count_ge"])
        plot_metric(
            df,
            metric,
            thresholds[metric],
            f"{args.plot_title}: {metric}",
            plot_dir / f"{metric}_by_condition.png",
        )
    (plot_dir / "dual_channel_threshold_summary.json").write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote dual-channel outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
