#!/usr/bin/env python3
"""Plot existing batch-aware positive-nucleus counts without reanalysing images."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


INPUT_CSV = Path("output/segmentation/batch_aware_current_dataset/comparison_field_replicates_single_dual_coloc.csv")
OUTPUT_DIR = Path("output/segmentation/batch_aware_current_dataset/plots_positive_nuclei_counts_2026-07-21")
EXPERIMENTS = ["TS III", "TS IV", "TS V"]
TIMEPOINTS = [2, 6, 24, 48]
CONDITION_ORDER = [
    "Aqueous",
    "DMSO",
    "NAH2PO4",
    "Media",
    "CX-5461",
    "Cisplatin",
    "Etoposide",
    "PDS",
    "Palbociclib",
]
PALETTE = ["#464C55", "#C5CAD3", "#7A828F", "#A3BEFA", "#F0986E", "#F390CA", "#A3D576", "#FFE15B"]
EDGES = ["#1F2430", "#7A828F", "#464C55", "#2E4780", "#804126", "#8A3A6F", "#386411", "#736422"]


def ordered_conditions(values: pd.Series) -> list[str]:
    present = set(values.dropna().astype(str))
    known = [condition for condition in CONDITION_ORDER if condition in present]
    return known + sorted(present - set(known))


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#E6E8F0", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)


def plot_metric(field: pd.DataFrame, metric: str, title: str, subtitle: str) -> plt.Figure:
    rng = np.random.default_rng(20260721)
    maximum = max(1, int(field[metric].max()))
    figure, axes = plt.subplots(3, 4, figsize=(20, 11.4), sharey=True, squeeze=False)
    figure.text(0.035, 0.985, title, ha="left", va="top", fontsize=14, fontweight="semibold")
    figure.text(0.035, 0.952, textwrap.fill(subtitle, width=145), ha="left", va="top", fontsize=9, color="#6F768A")

    for row, experiment in enumerate(EXPERIMENTS):
        for column, timepoint in enumerate(TIMEPOINTS):
            axis = axes[row, column]
            subset = field[(field["experiment"] == experiment) & (field["timepoint_hr"] == timepoint)]
            axis.set_title(f"{experiment}, {timepoint} hr", fontsize=10)
            if subset.empty:
                axis.text(0.5, 0.5, "not acquired", ha="center", va="center", transform=axis.transAxes, color="#6F768A")
                axis.set_xticks([])
                style_axis(axis)
                continue
            conditions = ordered_conditions(subset["condition"])
            for index, condition in enumerate(conditions):
                values = subset.loc[subset["condition"] == condition, metric].to_numpy(dtype=float)
                x_values = np.full(values.shape, index, dtype=float) + rng.uniform(-0.13, 0.13, size=values.shape)
                axis.scatter(
                    x_values,
                    values,
                    s=40,
                    color=PALETTE[index % len(PALETTE)],
                    edgecolor=EDGES[index % len(EDGES)],
                    linewidth=0.8,
                    alpha=0.9,
                    zorder=3,
                )
                axis.plot([index - 0.24, index + 0.24], [np.mean(values), np.mean(values)], color="#1F2430", linewidth=1.4)
            axis.set_xticks(range(len(conditions)))
            axis.set_xticklabels(conditions, rotation=45, ha="right")
            axis.set_xlim(-0.6, max(len(conditions) - 0.4, 0.6))
            axis.set_ylim(0, maximum * 1.12)
            if column == 0:
                axis.set_ylabel("Positive nuclei per CZI field")
            style_axis(axis)

    figure.text(0.035, 0.02, "Each dot is one CZI image field. Horizontal black ticks are condition/time field means. Counts are raw numerator counts; field nuclei totals vary.", fontsize=8, color="#6F768A")
    figure.tight_layout(rect=(0.02, 0.05, 0.995, 0.91))
    return figure


def main() -> None:
    field = pd.read_csv(INPUT_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    metrics = [
        (
            "n_single_af488_positive_batch",
            "AF488/53BP1-positive nuclei per field",
            "Threshold: TS III >=3 AF488 spots per nucleus; TS IV and TS V >=5.",
            "01_AF488_53BP1_positive_nuclei_per_field",
        ),
        (
            "n_single_rhrex_positive_batch",
            "RhReX putative gH2AX-positive nuclei per field",
            "Threshold: >=1 RhReX focus per nucleus in TS III, TS IV and TS V.",
            "02_RhReX_gH2AX_positive_nuclei_per_field",
        ),
        (
            "n_dual_channel_positive_batch",
            "Dual-channel-positive nuclei per field",
            "A nucleus is positive only when it meets both the AF488/53BP1 and RhReX putative gH2AX count thresholds. Spatial overlap is not required.",
            "03_dual_channel_positive_nuclei_per_field",
        ),
        (
            "n_coloc_positive_batch",
            "Spatially co-localisation-positive nuclei per field",
            "Threshold: >=1 RhReX focus spatially overlapping an AF488 focus after one-pixel AF488-mask dilation, within a nucleus.",
            "04_colocalized_positive_nuclei_per_field",
        ),
    ]
    with PdfPages(OUTPUT_DIR / "batch_aware_positive_nuclei_counts_field_replicates.pdf") as pdf:
        for metric, title, subtitle, stem in metrics:
            figure = plot_metric(field, metric, title, subtitle)
            figure.savefig(OUTPUT_DIR / f"{stem}.png", dpi=220)
            pdf.savefig(figure)
            plt.close(figure)
    pd.DataFrame(
        [{"metric": metric, "title": title, "definition": subtitle} for metric, title, subtitle, _ in metrics]
    ).to_csv(OUTPUT_DIR / "metric_definitions.csv", index=False)
    print(f"Wrote {len(metrics)} field-replicate count plots and one multi-page PDF to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
