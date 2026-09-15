#!/usr/bin/env python3
"""Create per-channel segmentation QC examples for dual-channel co-localisation."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import segmentation


DEFAULT_SEGMENTATION = Path(
    "/Users/dyap/Documents/Tasks workflow/czi-analysis/output/segmentation/"
    "time_series_IV_dual_channel_coloc"
)
DEFAULT_FILES = [
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


def display_u8(image: np.ndarray, low: float = 1.0, high: float = 99.8) -> np.ndarray:
    p_low, p_high = np.percentile(image, [low, high])
    if p_high <= p_low:
        p_low, p_high = float(image.min()), float(image.max())
    if p_high <= p_low:
        return np.zeros_like(image, dtype=np.uint8)
    return np.clip((image.astype(np.float32) - p_low) * 255.0 / (p_high - p_low), 0, 255).astype(np.uint8)


def mask_boundaries(labels: np.ndarray) -> np.ndarray:
    return segmentation.find_boundaries(labels, mode="outer")


def load_labels(segmentation_dir: Path, sample_id: str) -> dict[str, np.ndarray]:
    paths = {
        "nuclei": segmentation_dir / "masks" / f"{sample_id}_nuclei_labels.tif",
        "af488": segmentation_dir / "masks" / f"{sample_id}_af488_spot_labels.tif",
        "rhrex": segmentation_dir / "masks" / f"{sample_id}_rhrex_putative_gH2AX_labels.tif",
        "coloc": segmentation_dir / "masks" / f"{sample_id}_colocalized_rhrex_labels.tif",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing masks: " + ", ".join(missing))
    return {key: np.array(Image.open(path)) for key, path in paths.items()}


def overlay_boundaries(gray: np.ndarray, boundaries: list[tuple[np.ndarray, tuple[int, int, int]]]) -> np.ndarray:
    rgb = np.repeat(display_u8(gray)[..., None], 3, axis=2)
    for boundary, colour in boundaries:
        rgb[boundary] = colour
    return rgb


def binary_panel(base: np.ndarray, labels_a: np.ndarray, labels_b: np.ndarray, labels_coloc: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*base.shape, 3), dtype=np.uint8)
    rgb[:, :, :] = np.repeat(display_u8(base)[..., None] // 4, 3, axis=2)
    rgb[labels_a > 0] = np.array([0, 255, 255], dtype=np.uint8)
    rgb[labels_b > 0] = np.array([255, 128, 0], dtype=np.uint8)
    rgb[labels_coloc > 0] = np.array([255, 0, 255], dtype=np.uint8)
    return rgb


def make_figure(files: list[Path], output_png: Path, output_pdf: Path, segmentation_dir: Path) -> None:
    base = load_module("segment_base_for_channel_qc", "segment_nuclei_spots.py")
    inspector = base.load_inspector()

    fig, axes = plt.subplots(len(files), 4, figsize=(16, 4 * len(files)))
    if len(files) == 1:
        axes = np.array([axes])

    for row, czi in enumerate(files):
        inspection = inspector.inspect(czi, True, None)
        arrays = base.read_channel_arrays(czi, inspection, inspector)
        rhrex, af488, dapi = arrays[0], arrays[1], arrays[2]
        sid = base.sample_id(czi)
        condition, timepoint, replicate = base.parse_condition(czi)
        labels = load_labels(segmentation_dir, sid)

        panels = [
            (
                "DAPI-T3 nuclei segmentation\nraw DAPI + yellow nuclei boundary",
                overlay_boundaries(dapi, [(mask_boundaries(labels["nuclei"]), (255, 255, 0))]),
            ),
            (
                "AF488-T2 spot segmentation\nraw AF488 + cyan spot boundary",
                overlay_boundaries(af488, [(mask_boundaries(labels["af488"]), (0, 255, 255))]),
            ),
            (
                "RhReX-T1 putative gH2AX foci\nraw RhReX + orange focus boundary",
                overlay_boundaries(rhrex, [(mask_boundaries(labels["rhrex"]), (255, 128, 0))]),
            ),
            (
                "Co-localisation call\ncyan=AF488, orange=RhReX, magenta=co-localised RhReX",
                binary_panel(dapi, labels["af488"], labels["rhrex"], labels["coloc"]),
            ),
        ]

        for col, (title, image) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_axis_off()
            ax.set_title(title, fontsize=10)
            if col == 0:
                ax.text(
                    0,
                    -35,
                    f"{condition} | {timepoint} hr | rep {replicate}\n{czi.name}",
                    fontsize=10,
                    transform=ax.transData,
                    va="top",
                )

    fig.suptitle("Individual-channel segmentation QC examples", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("files", nargs="*", type=Path, default=DEFAULT_FILES)
    args = parser.parse_args()
    make_figure(args.files or DEFAULT_FILES, args.output_png, args.output_pdf, args.segmentation_dir)
    print(f"Wrote {args.output_png}")
    print(f"Wrote {args.output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
