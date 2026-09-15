#!/usr/bin/env python3
"""Plot pooled AF488 spots per nucleus and evaluate simple count cut-offs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
CONTROL_CONDITIONS = {"Aqueous", "DMSO", "NAH2PO4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="nucleus_measurements.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Pooled per-nucleus AF488 spot counts")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def jitter(values: np.ndarray, width: float = 0.28) -> np.ndarray:
    rng = np.random.default_rng(12345)
    return values + rng.uniform(-width, width, size=values.shape)


def threshold_table(df: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    treated = ~df["condition"].isin(CONTROL_CONDITIONS)
    controls = df["condition"].isin(CONTROL_CONDITIONS)
    max_count = int(df["spots_count"].max())
    for threshold in range(0, max_count + 2):
        positive = df["spots_count"] >= threshold
        tp = int((positive & treated).sum())
        fp = int((positive & controls).sum())
        tn = int((~positive & controls).sum())
        fn = int((~positive & treated).sum())
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        youden_j = sensitivity + specificity - 1.0
        rows.append(
            {
                "positive_if_spots_count_ge": threshold,
                "true_positive_treated_nuclei": tp,
                "false_positive_control_nuclei": fp,
                "true_negative_control_nuclei": tn,
                "false_negative_treated_nuclei": fn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "youden_j": youden_j,
                "control_positive_fraction": fp / int(controls.sum()) if controls.sum() else 0.0,
                "treated_positive_fraction": tp / int(treated.sum()) if treated.sum() else 0.0,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output, index=False)
    return table


def condition_summary(df: pd.DataFrame, output: Path) -> pd.DataFrame:
    grouped = df.groupby(["timepoint_hr", "condition"], sort=False, observed=True)
    rows = []
    for (timepoint, condition), group in grouped:
        rows.append(
            {
                "timepoint_hr": timepoint,
                "condition": condition,
                "n_nuclei": int(len(group)),
                "mean_spots_per_nucleus": float(group["spots_count"].mean()),
                "median_spots_per_nucleus": float(group["spots_count"].median()),
                "fraction_nuclei_gt3_spots": float((group["spots_count"] > 3).mean()),
                "fraction_nuclei_ge4_spots": float((group["spots_count"] >= 4).mean()),
                "p90_spots_per_nucleus": float(group["spots_count"].quantile(0.90)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output, index=False)
    return summary


def overall_condition_summary(df: pd.DataFrame, output: Path) -> pd.DataFrame:
    grouped = df.groupby(["condition"], sort=False, observed=True)
    rows = []
    for condition, group in grouped:
        if isinstance(condition, tuple):
            condition = condition[0]
        rows.append(
            {
                "condition": condition,
                "n_nuclei": int(len(group)),
                "mean_spots_per_nucleus": float(group["spots_count"].mean()),
                "median_spots_per_nucleus": float(group["spots_count"].median()),
                "fraction_nuclei_gt3_spots": float((group["spots_count"] > 3).mean()),
                "fraction_nuclei_ge3_spots": float((group["spots_count"] >= 3).mean()),
                "fraction_nuclei_ge4_spots": float((group["spots_count"] >= 4).mean()),
                "p90_spots_per_nucleus": float(group["spots_count"].quantile(0.90)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output, index=False)
    return summary


def make_plot(df: pd.DataFrame, threshold: int, title: str, output_png: Path, output_pdf: Path) -> None:
    colours = dict(zip(CONDITION_ORDER, plt.cm.tab10(np.linspace(0, 1, len(CONDITION_ORDER)))))
    timepoints = sorted(df["timepoint_hr"].unique(), key=lambda x: int(x))
    fig, axes = plt.subplots(1, len(timepoints), figsize=(16, 7), sharey=True)
    if len(timepoints) == 1:
        axes = [axes]
    for ax, timepoint in zip(axes, timepoints):
        sub = df[df["timepoint_hr"] == timepoint]
        for idx, condition in enumerate(CONDITION_ORDER):
            values = sub.loc[sub["condition"] == condition, "spots_count"].to_numpy(dtype=float)
            if values.size == 0:
                continue
            x = jitter(np.full(values.shape, idx, dtype=float))
            ax.scatter(
                x,
                values,
                s=10,
                alpha=0.45,
                color=colours[condition],
                edgecolors="none",
                label=condition if timepoint == timepoints[0] else None,
            )
            median = float(np.median(values))
            ax.plot([idx - 0.25, idx + 0.25], [median, median], color="black", linewidth=2)
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.3, label=f"candidate cut-off >= {threshold}" if ax is axes[0] else None)
        ax.set_title(f"{timepoint} hr")
        ax.set_xticks(range(len(CONDITION_ORDER)))
        ax.set_xticklabels(CONDITION_ORDER, rotation=45, ha="right")
        ax.set_xlabel("Condition")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("AF488 spots per DAPI-segmented nucleus")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(output_png, dpi=200)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df["timepoint_hr"] = df["timepoint_hr"].astype(str)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)
    df = df.sort_values(["timepoint_hr", "condition", "replicate", "sample_id", "nucleus_label"])

    summary = condition_summary(df, args.output_dir / "condition_timepoint_summary.csv")
    overall = overall_condition_summary(df, args.output_dir / "condition_summary_pooled.csv")
    thresholds = threshold_table(df, args.output_dir / "threshold_performance_controls_vs_treated.csv")
    for timepoint in sorted(df["timepoint_hr"].unique(), key=lambda x: int(x)):
        threshold_table(
            df[df["timepoint_hr"] == timepoint],
            args.output_dir / f"threshold_performance_controls_vs_treated_{timepoint}hr.csv",
        )
    best = thresholds.sort_values(["youden_j", "specificity", "sensitivity"], ascending=False).iloc[0]
    candidate = int(best["positive_if_spots_count_ge"])
    make_plot(
        df,
        candidate,
        args.title,
        args.output_dir / "af488_spots_per_nucleus_by_condition.png",
        args.output_dir / "af488_spots_per_nucleus_by_condition.pdf",
    )

    with (args.output_dir / "plot_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Input nuclei: {len(df)}\n")
        handle.write(f"Controls: {', '.join(sorted(CONTROL_CONDITIONS))}\n")
        handle.write("Treated: CX-5461, Cisplatin, Etoposide, PDS, Palbociclib\n")
        handle.write(f"Best simple integer cut-off by Youden J: spots_count >= {candidate}\n")
        handle.write(
            f"Sensitivity={best['sensitivity']:.4f}; specificity={best['specificity']:.4f}; "
            f"control positive fraction={best['control_positive_fraction']:.4f}; "
            f"treated positive fraction={best['treated_positive_fraction']:.4f}\n"
        )

    print(f"Wrote plots and threshold tables to {args.output_dir}")
    print(f"Best simple cut-off by Youden J: spots_count >= {candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
