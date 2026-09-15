"""Output writers for CZI foci analysis."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from skimage import segmentation

from .config import AnalysisConfig
from .segmentation import display_u8


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
    nuclear: np.ndarray,
    focus_a: np.ndarray,
    focus_b: np.ndarray,
    nuclei_labels: np.ndarray,
    focus_a_labels: np.ndarray,
    focus_b_labels: np.ndarray,
    coloc_focus_b_labels: np.ndarray,
    path: Path,
    config: AnalysisConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    low = config.output.qc_display_percentile_low
    high = config.output.qc_display_percentile_high
    rgb = np.zeros((*nuclear.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = display_u8(focus_b, low, high)
    rgb[:, :, 1] = display_u8(focus_a, low, high)
    rgb[:, :, 2] = display_u8(nuclear, low, high)

    rgb[segmentation.find_boundaries(nuclei_labels, mode="outer")] = np.array([255, 255, 0], dtype=np.uint8)
    rgb[segmentation.find_boundaries(focus_a_labels, mode="outer")] = np.array([0, 255, 255], dtype=np.uint8)
    rgb[segmentation.find_boundaries(focus_b_labels, mode="outer")] = np.array([255, 128, 0], dtype=np.uint8)
    rgb[segmentation.find_boundaries(coloc_focus_b_labels, mode="outer")] = np.array([255, 0, 255], dtype=np.uint8)

    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 5, 760, 78), fill=(0, 0, 0))
    draw.text((12, 12), f"Blue={config.nuclei.label}; Green={config.focus_a.label}; Red={config.focus_b.label}", fill=(255, 255, 255))
    draw.text((12, 31), f"Yellow=nuclei; cyan={config.focus_a.name}; orange={config.focus_b.name}", fill=(255, 255, 255))
    draw.text((12, 50), f"Magenta={config.focus_b.name} overlapping {config.focus_a.name} after {config.colocalization_dilation_px} px dilation", fill=(255, 255, 255))
    image.save(path)


def save_individual_channel_qc(
    image: np.ndarray,
    labels: np.ndarray,
    path: Path,
    channel_label: str,
    boundary_label: str,
    config: AnalysisConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    grey = display_u8(image, config.output.qc_display_percentile_low, config.output.qc_display_percentile_high)
    rgb = np.dstack([grey, grey, grey])
    rgb[segmentation.find_boundaries(labels, mode="outer")] = np.array([255, 0, 0], dtype=np.uint8)
    rendered = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(rendered)
    draw.rectangle((5, 5, 520, 40), fill=(0, 0, 0))
    draw.text((12, 14), f"{channel_label}; red boundary={boundary_label}", fill=(255, 255, 255))
    rendered.save(path)


def make_qc_pdf(
    path: Path,
    overlay_paths: list[Path],
    title: str,
    caption: str = "Composite QC: nuclear channel plus two focus channels and co-localised focus boundary.",
) -> None:
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
        pdf.drawString(36, height - 46, caption)
        pdf.drawRightString(width - 36, height - 46, f"Page {page_start // per_page + 1}")
        for index, overlay in enumerate(page_paths):
            col = index % 4
            row = index // 4
            x = 36 + col * 198
            y = height - 70 - row * 216 - thumb_h
            pdf.drawImage(ImageReader(str(overlay)), x, y, thumb_w, thumb_h, preserveAspectRatio=True, anchor="c")
            pdf.drawString(x, y - 11, overlay.stem.replace("_qc_overlay", ""))
        pdf.showPage()
    pdf.save()


def threshold_performance(
    nuclei: pd.DataFrame,
    metric: str,
    control_conditions: set[str],
    thresholds: list[int] | None = None,
) -> pd.DataFrame:
    controls = nuclei["condition"].isin(control_conditions)
    treated = ~controls
    max_count = int(nuclei[metric].max()) if len(nuclei) else 0
    candidate_thresholds = thresholds if thresholds is not None else list(range(max_count + 2))
    rows = []
    for threshold in candidate_thresholds:
        positive = nuclei[metric] >= threshold
        tp = int((positive & treated).sum())
        fp = int((positive & controls).sum())
        tn = int((~positive & controls).sum())
        fn = int((~positive & treated).sum())
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        rows.append(
            {
                "metric": metric,
                "positive_if_count_ge": int(threshold),
                "true_positive_treated_nuclei": tp,
                "false_positive_control_nuclei": fp,
                "true_negative_control_nuclei": tn,
                "false_negative_treated_nuclei": fn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "selectivity": specificity,
                "youden_j": sensitivity + specificity - 1.0,
                "control_positive_fraction": fp / int(controls.sum()) if controls.sum() else 0.0,
                "treated_positive_fraction": tp / int(treated.sum()) if treated.sum() else 0.0,
            }
        )
    return pd.DataFrame(rows)
