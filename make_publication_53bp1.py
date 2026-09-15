#!/usr/bin/env python3
"""Build publication-oriented 53BP1 figures from fixed-mask single-channel outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
PUB = ROOT / "output" / "publication_53BP1_gt5_v1"
DATA = ROOT / "output" / "publication_53BP1_v1" / "data"
FIGURES = PUB / "figures"
TABLES = PUB / "tables"

CONDITION_ORDER = ["Aqueous", "DMSO", "NAH2PO4", "Media", "CX-5461", "Cisplatin", "Etoposide", "PDS", "Palbociclib"]
CONTROLS = {"Aqueous", "DMSO", "NAH2PO4", "Media"}
POSITIVE_THRESHOLD = 5
TSV_MAP = {
    "A": "CX-5461", "B": "Cisplatin", "C": "Etoposide", "D": "Palbociclib",
    "E": "NAH2PO4", "F": "Media", "G": "DMSO", "H": "PDS",
}


def load_nuclei() -> pd.DataFrame:
    frames = []
    for experiment in ["TSIII", "TSIV", "TSV"]:
        frame = pd.read_csv(DATA / experiment / "nucleus_measurements.csv")
        frame["experiment"] = experiment
        if experiment == "TSV":
            frame["condition"] = frame["condition"].map(TSV_MAP)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_images() -> pd.DataFrame:
    frames = []
    for experiment in ["TSIII", "TSIV", "TSV"]:
        frame = pd.read_csv(DATA / experiment / "image_summary.csv")
        frame["experiment"] = experiment
        if experiment == "TSV":
            frame["condition"] = frame["condition"].map(TSV_MAP)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def field_table(nuclei: pd.DataFrame, images: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = images.loc[images["nuclei_count"] > 0].copy()
    excluded = images.loc[images["nuclei_count"] <= 0].copy()
    keys = ["experiment", "sample_id", "file_name", "condition", "timepoint_hr", "replicate"]
    fields = (
        nuclei.groupby(keys, as_index=False)
        .agg(
            nuclei_count=("nucleus_label", "size"),
            nuclei_gt5_53BP1=("53BP1_count", lambda x: int((x > POSITIVE_THRESHOLD).sum())),
            median_53BP1_foci_per_nucleus=("53BP1_count", "median"),
            mean_53BP1_foci_per_nucleus=("53BP1_count", "mean"),
            q1_53BP1_foci_per_nucleus=("53BP1_count", lambda x: x.quantile(0.25)),
            q3_53BP1_foci_per_nucleus=("53BP1_count", lambda x: x.quantile(0.75)),
        )
    )
    fields = fields.merge(valid[["experiment", "sample_id"]], on=["experiment", "sample_id"], how="inner")
    fields["percent_nuclei_gt5_53BP1"] = 100.0 * fields["nuclei_gt5_53BP1"] / fields["nuclei_count"]
    fields["condition"] = pd.Categorical(fields["condition"], CONDITION_ORDER, ordered=True)
    return fields.sort_values(["experiment", "timepoint_hr", "condition", "replicate"]), excluded


def plot_field_medians(fields: pd.DataFrame) -> tuple[Path, Path]:
    layout = {"TSIII": [2, 6], "TSIV": [2, 6, 24], "TSV": [24, 48]}
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 12.5), sharey=True)
    treatment_colour = "#087E8B"
    control_colour = "#747474"
    rng = np.random.default_rng(20260717)

    for row, (experiment, times) in enumerate(layout.items()):
        for col in range(3):
            ax = axes[row, col]
            if col >= len(times):
                ax.axis("off")
                continue
            timepoint = times[col]
            sub = fields[(fields["experiment"] == experiment) & (fields["timepoint_hr"] == timepoint)]
            present = [c for c in CONDITION_ORDER if c in set(sub["condition"].astype(str))]
            for x, condition in enumerate(present):
                values = sub.loc[sub["condition"].astype(str) == condition, "median_53BP1_foci_per_nucleus"].to_numpy(float)
                colour = control_colour if condition in CONTROLS else treatment_colour
                jitter = rng.uniform(-0.10, 0.10, len(values))
                ax.scatter(np.full(len(values), x) + jitter, values, s=34, facecolor=colour, edgecolor="white", linewidth=0.6, zorder=3)
                median = float(np.median(values))
                q1, q3 = np.quantile(values, [0.25, 0.75])
                ax.vlines(x, q1, q3, color="black", linewidth=1.3, zorder=4)
                ax.hlines(median, x - 0.20, x + 0.20, color="black", linewidth=2.2, zorder=5)
            ax.set_title(f"{experiment.replace('TS', 'TS ')} - {timepoint} h", loc="left", fontweight="bold")
            ax.set_xticks(range(len(present)))
            ax.set_xticklabels(present, rotation=42, ha="right")
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("Field median 53BP1 foci per nucleus")
    fig.suptitle("Nuclear 53BP1 foci burden by experiment and timepoint", fontsize=16, fontweight="bold", y=0.995)
    fig.text(0.5, 0.012, "Points are microscopy fields; horizontal line and vertical bar show median and IQR across fields.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.035, 1, 0.975))
    png = FIGURES / "field_median_53BP1_by_experiment_timepoint.png"
    pdf = FIGURES / "field_median_53BP1_by_experiment_timepoint.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_positive_fraction(fields: pd.DataFrame) -> tuple[Path, Path]:
    layout = {"TSIII": [2, 6], "TSIV": [2, 6, 24], "TSV": [24, 48]}
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 12.5), sharey=True)
    treatment_colour = "#087E8B"
    control_colour = "#747474"
    rng = np.random.default_rng(20260717)
    for row, (experiment, times) in enumerate(layout.items()):
        for col in range(3):
            ax = axes[row, col]
            if col >= len(times):
                ax.axis("off")
                continue
            timepoint = times[col]
            sub = fields[(fields["experiment"] == experiment) & (fields["timepoint_hr"] == timepoint)]
            present = [c for c in CONDITION_ORDER if c in set(sub["condition"].astype(str))]
            for x, condition in enumerate(present):
                values = sub.loc[sub["condition"].astype(str) == condition, "percent_nuclei_gt5_53BP1"].to_numpy(float)
                colour = control_colour if condition in CONTROLS else treatment_colour
                jitter = rng.uniform(-0.10, 0.10, len(values))
                ax.scatter(np.full(len(values), x) + jitter, values, s=34, facecolor=colour, edgecolor="white", linewidth=0.6, zorder=3)
                median = float(np.median(values))
                q1, q3 = np.quantile(values, [0.25, 0.75])
                ax.vlines(x, q1, q3, color="black", linewidth=1.3, zorder=4)
                ax.hlines(median, x - 0.20, x + 0.20, color="black", linewidth=2.2, zorder=5)
            ax.set_title(f"{experiment.replace('TS', 'TS ')} - {timepoint} h", loc="left", fontweight="bold")
            ax.set_xticks(range(len(present)))
            ax.set_xticklabels(present, rotation=42, ha="right")
            ax.set_ylim(-3, 103)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("Nuclei with >5 53BP1 foci (%)")
    fig.suptitle("53BP1-positive nuclei by experiment and timepoint", fontsize=16, fontweight="bold", y=0.995)
    fig.text(0.5, 0.012, "Points are microscopy fields; horizontal line and vertical bar show median and IQR across fields.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.035, 1, 0.975))
    png = FIGURES / "percent_nuclei_gt5_53BP1_by_experiment_timepoint.png"
    pdf = FIGURES / "percent_nuclei_gt5_53BP1_by_experiment_timepoint.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def representative_paths() -> list[tuple[str, str, Path]]:
    choices = [
        ("TS III - 6 h", "DMSO", DATA / "TSIII" / "qc_images" / "TSIII_2026-05-15_6hr_DMSO_rep01_53BP1_qc.png"),
        ("TS III - 6 h", "CX-5461", DATA / "TSIII" / "qc_images" / "TSIII_2026-05-15_6hr_CX_5461_rep01_53BP1_qc.png"),
        ("TS IV - 24 h", "DMSO", DATA / "TSIV" / "qc_images" / "TSIV_2026-05-26_24hr_DMSO_rep01_53BP1_qc.png"),
        ("TS IV - 24 h", "CX-5461", DATA / "TSIV" / "qc_images" / "TSIV_2026-05-26_24hr_CX_5461_rep01_53BP1_qc.png"),
        ("TS V - 48 h", "DMSO", DATA / "TSV" / "qc_images" / "TSV_2026-06-09_48hr_G_rep01_53BP1_qc.png"),
        ("TS V - 48 h", "CX-5461", DATA / "TSV" / "qc_images" / "TSV_2026-06-09_48hr_A_rep01_53BP1_qc.png"),
    ]
    for _, _, path in choices:
        if not path.is_file():
            raise FileNotFoundError(path)
    return choices


def plot_representative_qc() -> tuple[Path, Path]:
    choices = representative_paths()
    fig, axes = plt.subplots(3, 2, figsize=(10, 14))
    for ax, (experiment_time, condition, path) in zip(axes.flat, choices):
        ax.imshow(Image.open(path))
        ax.set_title(f"{experiment_time} | {condition}", loc="left", fontweight="bold")
        ax.axis("off")
    fig.suptitle("Representative 53BP1 focus-detection QC", fontsize=16, fontweight="bold", y=0.995)
    fig.text(0.5, 0.012, "AF488 channel; red boundaries indicate detected 53BP1 foci. Field 01 selected by a fixed rule.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.025, 1, 0.975))
    png = FIGURES / "representative_53BP1_qc.png"
    pdf = FIGURES / "representative_53BP1_qc.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    nuclei = load_nuclei()
    images = load_images()
    fields, excluded = field_table(nuclei, images)
    fields.to_csv(TABLES / "field_level_53BP1_summary.csv", index=False)
    nuclei.to_csv(TABLES / "combined_nucleus_53BP1_counts.csv", index=False)
    excluded.to_csv(TABLES / "excluded_fields_zero_approved_nuclei.csv", index=False)
    plot_field_medians(fields)
    plot_positive_fraction(fields)
    plot_representative_qc()
    (PUB / "README.md").write_text(
        "# Publication 53BP1 analysis v1\n\n"
        "Primary endpoint: percentage of accepted nuclei per field with more than 5 nuclear 53BP1 foci.\n"
        "Formula: 100 x nuclei(53BP1 count > 5) / accepted nuclei in field.\n\n"
        "- Approved nuclei masks were reused without re-segmentation.\n"
        "- 53BP1 was detected from AF488 only.\n"
        "- Gamma-H2AX and co-localisation were not analysed.\n"
        "- Each plotted point is a microscopy field.\n"
        "- The continuous field-median focus count is retained as a supplementary endpoint.\n"
        "- TS IV 24 hr NAH2PO4 field 05 was excluded because its approved mask contains zero nuclei.\n"
        "- Representative images use field 01 by a fixed rule, comparing DMSO with CX-5461 at the latest available timepoint in each experiment.\n",
        encoding="utf-8",
    )
    print(f"fields={len(fields)} nuclei={len(nuclei)} excluded_fields={len(excluded)}")


if __name__ == "__main__":
    main()
