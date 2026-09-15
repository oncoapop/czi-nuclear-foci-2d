#!/usr/bin/env python3
"""Create zoomed examples of AF488/RhReX co-localisation events."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import segmentation


SEGMENTATION_DIR = Path(
    "/Users/dyap/Documents/Tasks workflow/czi-analysis/output/segmentation/"
    "time_series_IV_dual_channel_coloc"
)
EXAMPLE_FILES = [
    Path("/Volumes/Backup/TS runs/Time Series IV (weird controls)/2026-05-26/24 hr 20x CX-5461-01.czi"),
    Path("/Volumes/Backup/TS runs/Time Series IV (weird controls)/2026-05-26/24 hr 20x NAH2PO4-01.czi"),
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


def display_u8(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [1, 99.8])
    if hi <= lo:
        lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    return np.clip((image.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def crop_bounds(center_y: float, center_x: float, shape: tuple[int, int], size: int = 220) -> tuple[slice, slice]:
    half = size // 2
    y0 = max(0, int(round(center_y)) - half)
    x0 = max(0, int(round(center_x)) - half)
    y1 = min(shape[0], y0 + size)
    x1 = min(shape[1], x0 + size)
    y0 = max(0, y1 - size)
    x0 = max(0, x1 - size)
    return slice(y0, y1), slice(x0, x1)


def load_labels(segmentation_dir: Path, sample_id: str) -> dict[str, np.ndarray]:
    names = {
        "nuclei": f"{sample_id}_nuclei_labels.tif",
        "af488": f"{sample_id}_af488_spot_labels.tif",
        "rhrex": f"{sample_id}_rhrex_putative_gH2AX_labels.tif",
        "coloc": f"{sample_id}_colocalized_rhrex_labels.tif",
    }
    return {key: np.array(Image.open(segmentation_dir / "masks" / name)) for key, name in names.items()}


def choose_nucleus(segmentation_dir: Path, sample_id: str) -> int:
    table = segmentation_dir / "nucleus_dual_channel_measurements.csv"
    candidates = []
    with table.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["sample_id"] != sample_id:
                continue
            coloc = int(row["colocalized_rhrex_spots_count"])
            if coloc <= 0:
                continue
            candidates.append((coloc, int(row["rhrex_spots_count"]), int(row["af488_spots_count"]), int(row["nucleus_label"])))
    if not candidates:
        raise ValueError(f"No co-localised nucleus found for {sample_id}")
    candidates.sort(reverse=True)
    return candidates[0][3]


def overlay(gray: np.ndarray, labels: list[tuple[np.ndarray, tuple[int, int, int]]]) -> np.ndarray:
    rgb = np.repeat(display_u8(gray)[..., None], 3, axis=2)
    for label_image, colour in labels:
        rgb[segmentation.find_boundaries(label_image, mode="outer")] = colour
    return rgb


def make_zoom(files: list[Path], output_png: Path, output_pdf: Path, segmentation_dir: Path) -> None:
    base = load_module("segment_base_for_coloc_zoom", "segment_nuclei_spots.py")
    inspector = base.load_inspector()
    fig, axes = plt.subplots(len(files), 5, figsize=(17, 4 * len(files)))
    if len(files) == 1:
        axes = np.array([axes])

    for row, czi in enumerate(files):
        inspection = inspector.inspect(czi, True, None)
        rhrex, af488, dapi = base.read_channel_arrays(czi, inspection, inspector)
        sid = base.sample_id(czi)
        condition, timepoint, replicate = base.parse_condition(czi)
        labels = load_labels(segmentation_dir, sid)
        nucleus_label = choose_nucleus(segmentation_dir, sid)
        ys, xs = np.where(labels["nuclei"] == nucleus_label)
        yslice, xslice = crop_bounds(float(ys.mean()), float(xs.mean()), dapi.shape)

        nuclei_crop = np.where(labels["nuclei"][yslice, xslice] == nucleus_label, nucleus_label, 0)
        af_crop = af488[yslice, xslice]
        rh_crop = rhrex[yslice, xslice]
        dapi_crop = dapi[yslice, xslice]
        af_lab = labels["af488"][yslice, xslice]
        rh_lab = labels["rhrex"][yslice, xslice]
        coloc_lab = labels["coloc"][yslice, xslice]

        composite = np.zeros((*dapi_crop.shape, 3), dtype=np.uint8)
        composite[:, :, 0] = display_u8(rh_crop)
        composite[:, :, 1] = display_u8(af_crop)
        composite[:, :, 2] = display_u8(dapi_crop)
        composite[segmentation.find_boundaries(coloc_lab, mode="outer")] = np.array([255, 0, 255], dtype=np.uint8)

        panels = [
            ("DAPI nucleus\nyellow=nucleus", overlay(dapi_crop, [(nuclei_crop, (255, 255, 0))])),
            ("AF488 spots\ncyan=AF488 mask", overlay(af_crop, [(af_lab, (0, 255, 255))])),
            ("RhReX foci\norange=RhReX mask", overlay(rh_crop, [(rh_lab, (255, 128, 0))])),
            ("Co-localised RhReX\nmagenta=co-localised focus", overlay(rh_crop, [(coloc_lab, (255, 0, 255))])),
            ("Merged crop\nR=RhReX G=AF488 B=DAPI", composite),
        ]
        for col, (title, image) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_axis_off()
            ax.set_title(title, fontsize=10)
            if col == 0:
                ax.text(
                    0,
                    -16,
                    f"{condition} {timepoint} hr rep {replicate}; nucleus {nucleus_label}",
                    fontsize=9,
                    transform=ax.transData,
                    va="top",
                )
    fig.suptitle("Zoomed examples of AF488/RhReX co-localisation calls", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", type=Path, default=SEGMENTATION_DIR)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("files", nargs="*", type=Path, default=EXAMPLE_FILES)
    args = parser.parse_args()
    make_zoom(args.files or EXAMPLE_FILES, args.output_png, args.output_pdf, args.segmentation_dir)
    print(f"Wrote {args.output_png}")
    print(f"Wrote {args.output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
