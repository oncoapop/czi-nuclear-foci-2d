"""Detect two focus channels while reusing fixed, previously approved nuclei masks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import measure

from .config import AnalysisConfig, config_to_dict
from .io import inspect_czi, load_condition_map, parse_sample_metadata, pixel_size_um, read_channel_arrays
from .reports import make_qc_pdf, save_individual_channel_qc, save_label_tiff, write_csv
from .segmentation import segment_spots
from .colocalization import spot_to_nucleus_labels


def _load_approved_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Approved nuclei mask not found: {path}")
    labels = np.asarray(Image.open(path))
    if labels.ndim != 2 or labels.shape != expected_shape:
        raise ValueError(
            f"Approved nuclei mask shape {labels.shape} does not match CZI plane {expected_shape}: {path}"
        )
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError(f"Approved nuclei mask must contain non-negative integer labels: {path}")
    return labels.astype(np.uint16, copy=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _focus_rows(labels: np.ndarray, intensity: np.ndarray, nuclei: np.ndarray, sample: dict, channel: str, scale: float) -> list[dict]:
    mapping = spot_to_nucleus_labels(labels, nuclei)
    rows = []
    for region in measure.regionprops(labels, intensity_image=intensity):
        rows.append(
            {
                **sample,
                "focus_channel": channel,
                "focus_label": int(region.label),
                "nucleus_label": mapping.get(int(region.label), 0),
                "focus_area_px": int(region.area),
                "focus_area_um2": float(region.area * scale),
                "focus_mean_intensity": float(region.mean_intensity),
                "focus_max_intensity": float(region.max_intensity),
                "focus_centroid_y_px": float(region.centroid[0]),
                "focus_centroid_x_px": float(region.centroid[1]),
            }
        )
    return rows


def _nucleus_rows(nuclei: np.ndarray, a_labels: np.ndarray, b_labels: np.ndarray, sample: dict, a_name: str, b_name: str) -> list[dict]:
    a_map = spot_to_nucleus_labels(a_labels, nuclei)
    b_map = spot_to_nucleus_labels(b_labels, nuclei)
    a_counts = {label: 0 for label in np.unique(nuclei) if label > 0}
    b_counts = dict(a_counts)
    for nucleus in a_map.values():
        if nucleus > 0:
            a_counts[nucleus] = a_counts.get(nucleus, 0) + 1
    for nucleus in b_map.values():
        if nucleus > 0:
            b_counts[nucleus] = b_counts.get(nucleus, 0) + 1
    return [
        {
            **sample,
            "nucleus_label": int(region.label),
            "nucleus_area_px": int(region.area),
            f"{a_name}_count_provisional": a_counts.get(int(region.label), 0),
            f"{b_name}_count_provisional": b_counts.get(int(region.label), 0),
        }
        for region in measure.regionprops(nuclei)
    ]


def analyse_focus_qc_with_masks(
    files: list[Path],
    config: AnalysisConfig,
    config_path: Path,
    nuclei_mask_dir: Path,
    output_dir: Path,
    qc_title: str,
) -> None:
    """Write focus-only QC products; deliberately make no co-localisation calls."""
    output_dir.mkdir(parents=True, exist_ok=False)
    condition_map_path = (config_path.parent / config.condition_map_csv).resolve() if config.condition_map_csv else None
    condition_map = load_condition_map(condition_map_path)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config_to_dict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest_rows: list[dict] = []
    image_rows: list[dict] = []
    nucleus_rows: list[dict] = []
    a_rows: list[dict] = []
    b_rows: list[dict] = []
    a_qc_paths: list[Path] = []
    b_qc_paths: list[Path] = []

    for czi_path in files:
        inspection = inspect_czi(czi_path)
        arrays = read_channel_arrays(czi_path, inspection)
        focus_a = arrays[config.focus_a.channel_index]
        focus_b = arrays[config.focus_b.channel_index]
        metadata = parse_sample_metadata(czi_path, config, condition_map)
        sample = metadata.__dict__
        mask_path = nuclei_mask_dir / f"{metadata.sample_id}_nuclei_labels.tif"
        nuclei = _load_approved_mask(mask_path, focus_a.shape)
        if focus_b.shape != focus_a.shape:
            raise ValueError(f"Focus channel shapes differ for {czi_path}")

        a_labels, a_diag = segment_spots(focus_a, nuclei, config.focus_a)
        b_labels, b_diag = segment_spots(focus_b, nuclei, config.focus_b)
        px_x, px_y = pixel_size_um(inspection["metadata"])
        scale = px_x * px_y if np.isfinite(px_x * px_y) else float("nan")
        sample_a = _focus_rows(a_labels, focus_a, nuclei, sample, config.focus_a.name, scale)
        sample_b = _focus_rows(b_labels, focus_b, nuclei, sample, config.focus_b.name, scale)
        sample_nuclei = _nucleus_rows(nuclei, a_labels, b_labels, sample, config.focus_a.name, config.focus_b.name)

        save_label_tiff(a_labels, output_dir / "focus_masks" / f"{metadata.sample_id}_{config.focus_a.name}_labels.tif")
        save_label_tiff(b_labels, output_dir / "focus_masks" / f"{metadata.sample_id}_{config.focus_b.name}_labels.tif")
        a_qc = output_dir / "qc_53BP1" / f"{metadata.sample_id}_{config.focus_a.name}_qc.png"
        b_qc = output_dir / "qc_gH2AX" / f"{metadata.sample_id}_{config.focus_b.name}_qc.png"
        save_individual_channel_qc(focus_a, a_labels, a_qc, config.focus_a.label, config.focus_a.name, config)
        save_individual_channel_qc(focus_b, b_labels, b_qc, config.focus_b.label, config.focus_b.name, config)
        a_qc_paths.append(a_qc)
        b_qc_paths.append(b_qc)

        manifest_rows.append(
            {
                **sample,
                "approved_nuclei_mask_path": str(mask_path),
                "approved_nuclei_mask_sha256": _sha256(mask_path),
                "image_height_px": int(focus_a.shape[0]),
                "image_width_px": int(focus_a.shape[1]),
                "nuclei_count": len(sample_nuclei),
            }
        )
        image_rows.append(
            {
                **sample,
                "nuclei_count": len(sample_nuclei),
                f"{config.focus_a.name}_count_provisional": len(sample_a),
                f"{config.focus_b.name}_count_provisional": len(sample_b),
                f"{config.focus_a.name}_threshold": a_diag[f"{config.focus_a.name}_threshold"],
                f"{config.focus_b.name}_threshold": b_diag[f"{config.focus_b.name}_threshold"],
            }
        )
        nucleus_rows.extend(sample_nuclei)
        a_rows.extend(sample_a)
        b_rows.extend(sample_b)
        print(f"{czi_path.name}: reused nuclei={len(sample_nuclei)} {config.focus_a.name}={len(sample_a)} {config.focus_b.name}={len(sample_b)}")

    write_csv(output_dir / "approved_nuclei_mask_manifest.csv", manifest_rows)
    write_csv(output_dir / "focus_qc_image_summary.csv", image_rows)
    write_csv(output_dir / "focus_qc_nucleus_counts_provisional.csv", nucleus_rows)
    write_csv(output_dir / f"{config.focus_a.name}_focus_measurements_provisional.csv", a_rows)
    write_csv(output_dir / f"{config.focus_b.name}_focus_measurements_provisional.csv", b_rows)
    make_qc_pdf(
        output_dir / f"{config.focus_a.name}_qc_contact_sheet.pdf",
        a_qc_paths,
        f"{qc_title}: {config.focus_a.label}",
        caption=f"Single-channel QC; red boundaries mark provisional {config.focus_a.name} foci.",
    )
    make_qc_pdf(
        output_dir / f"{config.focus_b.name}_qc_contact_sheet.pdf",
        b_qc_paths,
        f"{qc_title}: {config.focus_b.label}",
        caption=f"Single-channel QC; red boundaries mark provisional {config.focus_b.name} foci.",
    )
    (output_dir / "STATUS.txt").write_text(
        "FOCUS QC ONLY - PROVISIONAL\nNo co-localisation calls or biological summaries were calculated.\n",
        encoding="utf-8",
    )


def analyse_single_focus_with_masks(
    files: list[Path],
    config: AnalysisConfig,
    config_path: Path,
    nuclei_mask_dir: Path,
    output_dir: Path,
    qc_title: str,
) -> None:
    """Analyse focus A only using fixed nuclei labels; focus B is never read or segmented."""
    output_dir.mkdir(parents=True, exist_ok=False)
    condition_map_path = (config_path.parent / config.condition_map_csv).resolve() if config.condition_map_csv else None
    condition_map = load_condition_map(condition_map_path)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config_to_dict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_rows: list[dict] = []
    image_rows: list[dict] = []
    nucleus_rows: list[dict] = []
    focus_rows: list[dict] = []
    qc_paths: list[Path] = []

    for czi_path in files:
        inspection = inspect_czi(czi_path)
        arrays = read_channel_arrays(czi_path, inspection)
        focus = arrays[config.focus_a.channel_index]
        metadata = parse_sample_metadata(czi_path, config, condition_map)
        sample = metadata.__dict__
        mask_path = nuclei_mask_dir / f"{metadata.sample_id}_nuclei_labels.tif"
        nuclei = _load_approved_mask(mask_path, focus.shape)
        labels, diagnostics = segment_spots(focus, nuclei, config.focus_a)
        px_x, px_y = pixel_size_um(inspection["metadata"])
        scale = px_x * px_y if np.isfinite(px_x * px_y) else float("nan")
        sample_focus = _focus_rows(labels, focus, nuclei, sample, config.focus_a.name, scale)
        mapping = spot_to_nucleus_labels(labels, nuclei)
        counts: dict[int, int] = {}
        for nucleus in mapping.values():
            if nucleus > 0:
                counts[nucleus] = counts.get(nucleus, 0) + 1
        sample_nuclei = [
            {
                **sample,
                "nucleus_label": int(region.label),
                "nucleus_area_px": int(region.area),
                "nucleus_area_um2": float(region.area * scale),
                f"{config.focus_a.name}_count": counts.get(int(region.label), 0),
            }
            for region in measure.regionprops(nuclei)
        ]
        save_label_tiff(labels, output_dir / "focus_masks" / f"{metadata.sample_id}_{config.focus_a.name}_labels.tif")
        qc_path = output_dir / "qc_images" / f"{metadata.sample_id}_{config.focus_a.name}_qc.png"
        save_individual_channel_qc(focus, labels, qc_path, config.focus_a.label, config.focus_a.name, config)
        qc_paths.append(qc_path)
        manifest_rows.append(
            {
                **sample,
                "approved_nuclei_mask_path": str(mask_path),
                "approved_nuclei_mask_sha256": _sha256(mask_path),
                "nuclei_count": len(sample_nuclei),
                "image_height_px": int(focus.shape[0]),
                "image_width_px": int(focus.shape[1]),
            }
        )
        image_rows.append(
            {
                **sample,
                "nuclei_count": len(sample_nuclei),
                f"{config.focus_a.name}_total_count": len(sample_focus),
                f"{config.focus_a.name}_median_per_nucleus": float(np.median([r[f"{config.focus_a.name}_count"] for r in sample_nuclei])),
                f"{config.focus_a.name}_threshold": diagnostics[f"{config.focus_a.name}_threshold"],
            }
        )
        nucleus_rows.extend(sample_nuclei)
        focus_rows.extend(sample_focus)
        print(f"{czi_path.name}: reused nuclei={len(sample_nuclei)} {config.focus_a.name}={len(sample_focus)}")

    write_csv(output_dir / "approved_nuclei_mask_manifest.csv", manifest_rows)
    write_csv(output_dir / "image_summary.csv", image_rows)
    write_csv(output_dir / "nucleus_measurements.csv", nucleus_rows)
    write_csv(output_dir / f"{config.focus_a.name}_focus_measurements.csv", focus_rows)
    make_qc_pdf(
        output_dir / f"{config.focus_a.name}_qc_contact_sheet.pdf",
        qc_paths,
        qc_title,
        caption=f"Single-channel QC; red boundaries mark {config.focus_a.name} foci. Approved nuclei masks were reused.",
    )
    (output_dir / "STATUS.txt").write_text(
        f"SINGLE-CHANNEL {config.focus_a.name} ANALYSIS\nFocus B and co-localisation were not calculated.\n",
        encoding="utf-8",
    )
