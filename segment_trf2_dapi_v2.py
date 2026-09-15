#!/usr/bin/env python3
"""Revised TRF2/DAPI analysis with smoother nuclei and more sensitive puncta calls."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max

import segment_trf2_dapi as base


DEFAULT_ROOT = Path("/Volumes/Backup/TS Runs/TRF2Opt")
DEFAULT_OUTPUT = Path("output/trf2opt_trf2_dapi_v2")


@dataclass(frozen=True)
class V2Parameters(base.Parameters):
    nuclei_background_sigma_px: float = 24.0
    nuclei_opening_radius_px: int = 1
    nuclei_closing_radius_px: int = 2
    nuclei_watershed_min_distance_px: int = 14
    trf2_puncta_tophat_radius_px: int = 3
    trf2_puncta_threshold_mad_multiplier: float = 4.5
    trf2_min_intensity_above_local_background: float = 4.0
    trf2_puncta_max_area_px: int = 80
    trf2_puncta_min_mean_enhanced_intensity: float = 2.0


def background_correct(image: np.ndarray, sigma_px: float) -> np.ndarray:
    image_f = image.astype(np.float32)
    background = filters.gaussian(image_f, sigma=sigma_px, preserve_range=True)
    corrected = image_f - background
    corrected -= float(corrected.min())
    return corrected


def segment_nuclei_v2(
    dapi: np.ndarray,
    px_um_x: float,
    px_um_y: float,
    params: V2Parameters,
) -> tuple[np.ndarray, dict]:
    corrected = background_correct(dapi, params.nuclei_background_sigma_px)
    smoothed = filters.gaussian(corrected, sigma=params.nuclei_gaussian_sigma_px, preserve_range=True)
    threshold = float(filters.threshold_otsu(smoothed)) if np.unique(smoothed).size > 1 else float(smoothed.mean())
    min_area = base.area_um2_to_px(params.nuclei_min_area_um2, px_um_x, px_um_y)
    hole_area = base.area_um2_to_px(params.nuclei_hole_area_um2, px_um_x, px_um_y)

    binary = smoothed > threshold
    binary = morphology.opening(binary, morphology.disk(params.nuclei_opening_radius_px))
    binary = morphology.closing(binary, morphology.disk(params.nuclei_closing_radius_px))
    binary = base.remove_objects_smaller_than(binary, min_area)
    binary = base.remove_holes_smaller_than(binary, hole_area)
    binary = ndi.binary_fill_holes(binary)

    distance = filters.gaussian(ndi.distance_transform_edt(binary), sigma=1.0, preserve_range=True)
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
    labels = base.remove_objects_smaller_than(labels, min_area)
    labels = measure.label(labels > 0).astype(np.uint16)
    return labels, {
        "segmentation_profile": "v2_background_corrected_smooth_nuclei",
        "dapi_corrected_otsu_threshold": threshold,
        "nuclei_min_area_px": int(min_area),
        "nuclei_hole_area_px": int(hole_area),
        "nuclei_watershed_markers": int(markers.max()),
        "nuclei_count": int(labels.max()),
    }


def filter_puncta_labels(
    labels: np.ndarray,
    enhanced: np.ndarray,
    params: V2Parameters,
) -> np.ndarray:
    keep = np.zeros(labels.shape, dtype=bool)
    for region in measure.regionprops(labels, intensity_image=enhanced):
        if region.area > params.trf2_puncta_max_area_px:
            continue
        if base.intensity_mean(region) < params.trf2_puncta_min_mean_enhanced_intensity:
            continue
        keep[labels == region.label] = True
    return measure.label(keep).astype(np.uint16)


def segment_trf2_puncta_v2(
    trf2: np.ndarray,
    nuclei_labels: np.ndarray,
    params: V2Parameters,
) -> tuple[np.ndarray, np.ndarray, dict]:
    nuclear_mask = nuclei_labels > 0
    footprint = morphology.disk(params.trf2_puncta_tophat_radius_px)
    enhanced = morphology.white_tophat(trf2, footprint=footprint)
    threshold = base.robust_threshold(enhanced[nuclear_mask], params.trf2_puncta_threshold_mad_multiplier)
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
    binary = base.remove_objects_smaller_than(binary, params.trf2_puncta_min_area_px)
    labels = measure.label(binary).astype(np.uint16)
    labels = filter_puncta_labels(labels, enhanced, params)
    return labels, enhanced, {
        "segmentation_profile": "v2_background_corrected_smooth_nuclei",
        "trf2_puncta_tophat_threshold": float(threshold),
        "trf2_puncta_min_area_px": int(params.trf2_puncta_min_area_px),
        "trf2_puncta_max_area_px": int(params.trf2_puncta_max_area_px),
        "trf2_puncta_count": int(labels.max()),
    }


def write_method_notes(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# TRF2/DAPI v2 analysis notes",
                "",
                "This revised pass keeps the same verified channel assignment as v1:",
                "- C0: Rhodamine Red-X, used as TRF2.",
                "- C1: DAPI, used for nuclear segmentation.",
                "",
                "Changes from v1:",
                "- DAPI nuclei are background-corrected before Otsu thresholding.",
                "- Nuclear masks are lightly opened and closed to reduce jagged boundaries.",
                "- Watershed seeds use a larger minimum distance to reduce over-splitting.",
                "- TRF2 puncta threshold is more sensitive but remains gated by local background and max area.",
                "",
                "Interpretation boundary: TRF2 puncta-like objects are not independently confirmed telomere ends in this two-channel dataset.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def analyse(root: Path, output_dir: Path, params: V2Parameters) -> None:
    files = base.selected_files(root)
    if not files:
        raise ValueError(f"No matching CZI files found under {root}")

    inspector = base.load_inspector()
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
        sample = base.parse_sample(path)
        inspection = inspector.inspect(path, include_stats=False, preview_dir=None, plane_dir=None)
        arrays = base.read_channel_arrays(path, inspection, inspector)
        trf2 = arrays[params.trf2_channel_index]
        dapi = arrays[params.dapi_channel_index]
        px_um_x, px_um_y = base.pixel_size_um(inspection["metadata"])

        nuclei_labels, nuclei_diag = segment_nuclei_v2(dapi, px_um_x, px_um_y, params)
        puncta_labels, enhanced, puncta_diag = segment_trf2_puncta_v2(trf2, nuclei_labels, params)
        nucleus_rows, puncta_rows = base.measurement_rows(
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
        base.save_label_tiff(nuclei_labels, output_dir / "masks" / f"{sid}_nuclei_labels.tif")
        base.save_label_tiff(puncta_labels, output_dir / "masks" / f"{sid}_trf2_puncta_labels.tif")
        overlay_path = output_dir / "qc_overlays" / f"{sid}_qc_overlay.png"
        base.save_qc_overlay(dapi, trf2, nuclei_labels, puncta_labels, overlay_path, params)
        overlay_paths.append(overlay_path)

        image_rows.append(
            base.image_summary_row(
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
        metadata_rows.append(base.source_metadata_row(path, inspection, params))
        valid_nuclei = sum(1 for row in nucleus_rows if row["valid_for_summary"] == "TRUE")
        print(f"{path.name}: nuclei={len(nucleus_rows)} valid={valid_nuclei} trf2_puncta={len(puncta_rows)}")

    base.write_csv(output_dir / "source_metadata.csv", metadata_rows)
    base.write_csv(output_dir / "image_summary.csv", image_rows)
    base.write_csv(output_dir / "nucleus_measurements.csv", nucleus_rows_all)
    base.write_csv(output_dir / "trf2_puncta_measurements.csv", puncta_rows_all)
    base.write_condition_summary(output_dir / "condition_summary.csv", nucleus_rows_all)
    base.make_qc_pdf(output_dir / "qc_overlay_contact_sheet.pdf", overlay_paths, "TRF2Opt TRF2/DAPI v2 QC")
    print(f"Wrote revised TRF2/DAPI analysis outputs to {output_dir}")


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
        analyse(args.root, output_dir, V2Parameters())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
