#!/usr/bin/env python3
"""Plot labelled TRF2Opt v2 results without requiring matplotlib."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


DEFAULT_INPUT = Path("output/trf2opt_trf2_dapi_v2_labelled")
DEFAULT_OUTPUT = DEFAULT_INPUT / "plots"

FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
BOLD_FONT = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")

COLOURS = {
    "1 uM CX5461": (213, 94, 0),
    "NaH2PO4": (0, 114, 178),
    "None": (100, 100, 100),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(BOLD_FONT if bold else FONT), size=size)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def condition_colour(row: pd.Series) -> tuple[int, int, int]:
    if str(row["trf2_dilution"]) == "None":
        return COLOURS["None"]
    return COLOURS.get(row["drug"], (50, 50, 50))


def condition_label(row: pd.Series) -> str:
    treatment = "Treated" if row["drug"] == "1 uM CX5461" else "Untreated"
    dilution = "No Ab" if str(row["trf2_dilution"]) == "None" else str(row["trf2_dilution"])
    return f"{int(row['timepoint_hr'])} h\n{treatment}\n{dilution}"


def ordered_conditions(sample_sheet: pd.DataFrame) -> pd.DataFrame:
    sheet = sample_sheet.copy()
    sheet["trf2_dilution"] = sheet["trf2_dilution"].fillna("None").astype(str)
    sheet["drug_order"] = sheet["drug"].map({"NaH2PO4": 0, "1 uM CX5461": 1}).fillna(9)
    sheet["dilution_order"] = sheet["trf2_dilution"].map(
        {"1:300": 0, "1:500": 1, "1:1000": 2, "1:2000": 3, "None": 4}
    ).fillna(9)
    return (
        sheet.sort_values(["timepoint_hr", "drug_order", "dilution_order", "well"])
        .drop_duplicates(["timepoint_hr", "drug", "trf2_dilution"])
        .reset_index(drop=True)
    )


def deterministic_jitter(index: int, width: float) -> float:
    value = ((index * 1103515245 + 12345) & 0x7FFFFFFF) / 0x7FFFFFFF
    return (value - 0.5) * width


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, mean, mean
    sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def y_ticks(y_max: float, n_ticks: int = 6) -> list[float]:
    if y_max <= 10:
        step = 2
    elif y_max <= 50:
        step = 10
    else:
        step = 25
    top = math.ceil(y_max / step) * step
    ticks = [float(i) for i in np.arange(0, top + step, step)]
    if len(ticks) <= n_ticks:
        return ticks
    stride = math.ceil((len(ticks) - 1) / (n_ticks - 1))
    reduced = ticks[::stride]
    if reduced[-1] != ticks[-1]:
        reduced.append(ticks[-1])
    return reduced


def draw_distribution_plot(
    nuclei: pd.DataFrame,
    sample_sheet: pd.DataFrame,
    metric: str,
    y_label: str,
    title: str,
    output: Path,
    y_max: float | None = None,
) -> None:
    conditions = ordered_conditions(sample_sheet)
    width, height = 2300, 1250
    left, right, top, bottom = 130, 2220, 130, 920
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(36, True)
    axis_font = font(22)
    tick_font = font(20)
    label_font = font(18)
    small_font = font(16)

    draw.text((left, 35), title, fill=(20, 20, 20), font=title_font)
    draw.text((left, 82), "Each faint point is one valid non-border nucleus; thick marker is mean +/- 95% CI.", fill=(70, 70, 70), font=axis_font)

    values_all = nuclei[metric].to_numpy(dtype=float)
    if y_max is None:
        y_max = max(float(np.nanmax(values_all)) * 1.08, 1.0)
    y_min = 0.0

    def y_to_px(value: float) -> int:
        value = max(y_min, min(y_max, value))
        return int(bottom - ((value - y_min) / (y_max - y_min)) * (bottom - top))

    ticks = y_ticks(y_max)
    for tick in ticks:
        y = y_to_px(float(tick))
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
        draw.text((left - 78, y - 12), f"{tick:g}", fill=(60, 60, 60), font=tick_font)
    draw.line((left, bottom, right, bottom), fill=(40, 40, 40), width=2)
    draw.line((left, top, left, bottom), fill=(40, 40, 40), width=2)

    label_img = Image.new("RGBA", (420, 40), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((0, 0), y_label, fill=(35, 35, 35), font=axis_font)
    rotated = label_img.rotate(90, expand=True)
    image.paste(rotated, (25, top + 180), rotated)

    xs = np.linspace(left + 45, right - 45, len(conditions))
    previous_time = None
    for idx, (_, cond) in enumerate(conditions.iterrows()):
        x = float(xs[idx])
        if previous_time is not None and cond["timepoint_hr"] != previous_time:
            midpoint = (xs[idx - 1] + x) / 2
            draw.line((midpoint, top, midpoint, bottom + 105), fill=(180, 180, 180), width=2)
        previous_time = cond["timepoint_hr"]

        mask = (
            nuclei["timepoint_hr"].eq(cond["timepoint_hr"])
            & nuclei["drug"].eq(cond["drug"])
            & nuclei["trf2_dilution"].astype(str).eq(str(cond["trf2_dilution"]))
        )
        group = nuclei.loc[mask, metric].to_numpy(dtype=float)
        colour = condition_colour(cond)
        point_overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        point_draw = ImageDraw.Draw(point_overlay)
        for point_idx, value in enumerate(group):
            px = int(x + deterministic_jitter(point_idx, 52))
            py = y_to_px(float(value))
            point_draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(*colour, 45))
        image = Image.alpha_composite(image.convert("RGBA"), point_overlay).convert("RGB")
        draw = ImageDraw.Draw(image)

        if len(group):
            mean, lo, hi = mean_ci(group)
            y_mean, y_lo, y_hi = y_to_px(mean), y_to_px(lo), y_to_px(hi)
            draw.line((x, y_hi, x, y_lo), fill=(0, 0, 0), width=3)
            draw.line((x - 11, y_hi, x + 11, y_hi), fill=(0, 0, 0), width=3)
            draw.line((x - 11, y_lo, x + 11, y_lo), fill=(0, 0, 0), width=3)
            draw.ellipse((x - 10, y_mean - 10, x + 10, y_mean + 10), fill=colour, outline=(0, 0, 0), width=2)
            draw.text((x - 18, bottom + 8), f"n={len(group)}", fill=(70, 70, 70), font=small_font)

        for line_idx, line in enumerate(condition_label(cond).split("\n")):
            tw, _ = text_size(draw, line, label_font)
            draw.text((x - tw / 2, bottom + 35 + line_idx * 23), line, fill=(30, 30, 30), font=label_font)

    legend_x, legend_y = right - 520, 42
    legend_items = [
        ("Treated: 1 uM CX5461, Ab present", COLOURS["1 uM CX5461"]),
        ("Untreated: NaH2PO4, Ab present", COLOURS["NaH2PO4"]),
        ("No primary Ab control", COLOURS["None"]),
    ]
    for i, (text, colour) in enumerate(legend_items):
        y = legend_y + i * 30
        draw.rectangle((legend_x, y + 5, legend_x + 24, y + 21), fill=colour)
        draw.text((legend_x + 34, y), text, fill=(30, 30, 30), font=small_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def draw_ab_control_plot(nuclei: pd.DataFrame, output: Path) -> None:
    width, height = 1800, 950
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(34, True)
    axis_font = font(21)
    label_font = font(18)
    draw.text((80, 35), "TRF2 staining versus no-Ab controls", fill=(20, 20, 20), font=title_font)
    draw.text((80, 80), "Bars show mean across valid nuclei; error bars show 95% CI.", fill=(70, 70, 70), font=axis_font)

    panels = [
        ("nucleus_background_subtracted_mean_trf2", "Bg-subtracted nuclear TRF2", 90, 830),
        ("trf2_puncta_count", "TRF2 puncta-like objects/nucleus", 980, 1720),
    ]
    comparison_rows = []
    for (timepoint, drug, ab_state), group in nuclei.groupby(["timepoint_hr", "drug", "antibody_state"], sort=True):
        comparison_rows.append(
            {
                "timepoint_hr": timepoint,
                "drug": drug,
                "antibody_state": ab_state,
                "label": f"{int(timepoint)} h\n{'Treated' if drug == '1 uM CX5461' else 'Untreated'}\n{ab_state}",
                "n": len(group),
                "nucleus_background_subtracted_mean_trf2": group["nucleus_background_subtracted_mean_trf2"].to_numpy(float),
                "trf2_puncta_count": group["trf2_puncta_count"].to_numpy(float),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["order"] = comparisons["timepoint_hr"].astype(int) * 100 + comparisons["drug"].map({"NaH2PO4": 0, "1 uM CX5461": 1}) * 10 + comparisons["antibody_state"].map({"Ab": 0, "No Ab": 1})
    comparisons = comparisons.sort_values("order").reset_index(drop=True)

    for metric, panel_title, left, right in panels:
        top, bottom = 170, 700
        max_value = max(float(np.max(values)) for values in comparisons[metric])
        y_max = max_value * 1.15 if max_value else 1.0

        def y_to_px(value: float) -> int:
            return int(bottom - (value / y_max) * (bottom - top))

        draw.text((left, 130), panel_title, fill=(20, 20, 20), font=axis_font)
        for tick in y_ticks(y_max, 5):
            y = y_to_px(float(tick))
            draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
            draw.text((left - 60, y - 10), f"{tick:g}", fill=(70, 70, 70), font=label_font)
        draw.line((left, bottom, right, bottom), fill=(40, 40, 40), width=2)
        draw.line((left, top, left, bottom), fill=(40, 40, 40), width=2)

        xs = np.linspace(left + 55, right - 55, len(comparisons))
        for idx, row in comparisons.iterrows():
            values = row[metric]
            mean, lo, hi = mean_ci(values)
            colour = COLOURS["None"] if row["antibody_state"] == "No Ab" else COLOURS[row["drug"]]
            x = int(xs[idx])
            y_mean, y_lo, y_hi = y_to_px(mean), y_to_px(lo), y_to_px(hi)
            draw.rectangle((x - 18, y_mean, x + 18, bottom), fill=colour, outline=(0, 0, 0))
            draw.line((x, y_hi, x, y_lo), fill=(0, 0, 0), width=3)
            draw.line((x - 10, y_hi, x + 10, y_hi), fill=(0, 0, 0), width=3)
            draw.line((x - 10, y_lo, x + 10, y_lo), fill=(0, 0, 0), width=3)
            for line_idx, line in enumerate(str(row["label"]).split("\n")):
                tw, _ = text_size(draw, line, label_font)
                draw.text((x - tw / 2, bottom + 20 + line_idx * 22), line, fill=(30, 30, 30), font=label_font)
            draw.text((x - 18, bottom + 90), f"n={int(row['n'])}", fill=(80, 80, 80), font=label_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def make_pdf(image_paths: list[Path], output: Path) -> None:
    pdf = canvas.Canvas(str(output), pagesize=landscape(A4))
    width, height = landscape(A4)
    for image_path in image_paths:
        pdf.drawImage(
            ImageReader(str(image_path)),
            30,
            30,
            width - 60,
            height - 60,
            preserveAspectRatio=True,
            anchor="c",
        )
        pdf.showPage()
    pdf.save()


def write_plot_summary(nuclei: pd.DataFrame, output: Path) -> None:
    summary = (
        nuclei.groupby(["timepoint_hr", "drug", "trf2_dilution"], dropna=False)
        .agg(
            n_valid_nuclei=("nucleus_label", "count"),
            mean_background_subtracted_trf2=("nucleus_background_subtracted_mean_trf2", "mean"),
            mean_puncta_per_nucleus=("trf2_puncta_count", "mean"),
        )
        .reset_index()
        .sort_values(["timepoint_hr", "drug", "trf2_dilution"])
    )
    header = [
        "timepoint_hr",
        "drug",
        "trf2_dilution",
        "n_valid_nuclei",
        "mean_background_subtracted_trf2",
        "mean_puncta_per_nucleus",
    ]
    table = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for _, row in summary.iterrows():
        table.append(
            "| "
            + " | ".join(
                [
                    str(row["timepoint_hr"]),
                    str(row["drug"]),
                    str(row["trf2_dilution"]),
                    str(int(row["n_valid_nuclei"])),
                    f"{row['mean_background_subtracted_trf2']:.2f}",
                    f"{row['mean_puncta_per_nucleus']:.2f}",
                ]
            )
            + " |"
        )

    lines = [
        "# TRF2Opt v2 graph summary",
        "",
        "Graphs use valid non-border nuclei from the v2 segmentation.",
        "TRF2 None wells are shown as no-primary antibody controls.",
        "",
        *table,
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    nuclei = pd.read_csv(input_dir / "nucleus_measurements_labelled.csv")
    sample_sheet = pd.read_csv(input_dir / "sample_sheet_mapping.csv")
    nuclei["valid_for_summary"] = nuclei["valid_for_summary"].astype(str).str.upper().eq("TRUE")
    nuclei = nuclei[nuclei["valid_for_summary"]].copy()
    nuclei["trf2_dilution"] = nuclei["trf2_dilution"].fillna("None").astype(str)
    sample_sheet["trf2_dilution"] = sample_sheet["trf2_dilution"].fillna("None").astype(str)
    nuclei["antibody_state"] = np.where(nuclei["trf2_dilution"].eq("None"), "No Ab", "Ab")

    intensity_plot = output_dir / "treated_vs_untreated_nuclear_trf2.png"
    puncta_plot = output_dir / "treated_vs_untreated_trf2_puncta.png"
    control_plot = output_dir / "ab_control_check.png"

    draw_distribution_plot(
        nuclei,
        sample_sheet,
        "nucleus_background_subtracted_mean_trf2",
        "Background-subtracted mean TRF2 intensity",
        "Treated versus untreated: nuclear TRF2 staining",
        intensity_plot,
        y_max=180,
    )
    draw_distribution_plot(
        nuclei,
        sample_sheet,
        "trf2_puncta_count",
        "TRF2 puncta-like objects per nucleus",
        "Treated versus untreated: TRF2 puncta-like objects",
        puncta_plot,
        y_max=25,
    )
    draw_ab_control_plot(nuclei, control_plot)
    make_pdf([intensity_plot, puncta_plot, control_plot], output_dir / "trf2opt_v2_graphs.pdf")
    write_plot_summary(nuclei, output_dir / "graph_summary.md")
    print(f"Wrote graphs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
