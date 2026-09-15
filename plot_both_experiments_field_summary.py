#!/usr/bin/env python3
"""Create field-level summary plots for TS III and TS IV CZI analyses.

The replicate unit in this script is one acquired CZI image field, not one
nucleus. Raw CZI files are not read or modified; inputs are the existing
segmentation CSVs produced by the earlier analysis scripts.
"""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
TIMEPOINT_ORDER = [2, 6, 24]
EXPERIMENT_ORDER = ["TS III", "TS IV"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

CONDITION_COLOURS = {
    "Aqueous": "#464C55",
    "DMSO": "#C5CAD3",
    "NAH2PO4": "#7A828F",
    "CX-5461": "#A3BEFA",
    "Cisplatin": "#F0986E",
    "Etoposide": "#F390CA",
    "PDS": "#A3D576",
    "Palbociclib": "#FFE15B",
}

CONDITION_EDGES = {
    "Aqueous": "#1F2430",
    "DMSO": "#7A828F",
    "NAH2PO4": "#464C55",
    "CX-5461": "#2E4780",
    "Cisplatin": "#804126",
    "Etoposide": "#8A3A6F",
    "PDS": "#386411",
    "Palbociclib": "#736422",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tsiii-nuclei",
        type=Path,
        default=Path("output/segmentation/time_series_III_all/nucleus_measurements.csv"),
    )
    parser.add_argument(
        "--tsiii-images",
        type=Path,
        default=Path("output/segmentation/time_series_III_all/image_summary.csv"),
    )
    parser.add_argument(
        "--tsiv-nuclei",
        type=Path,
        default=Path("output/segmentation/time_series_IV_dual_channel_coloc/nucleus_dual_channel_measurements.csv"),
    )
    parser.add_argument(
        "--tsiv-images",
        type=Path,
        default=Path("output/segmentation/time_series_IV_dual_channel_coloc/image_summary_dual_channel.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/segmentation/summary_both_experiments"),
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.titlecolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 9,
        }
    )


def add_figure_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    title_text = textwrap.fill(title, width=95, break_long_words=False)
    subtitle_text = textwrap.fill(subtitle, width=130, break_long_words=False)
    fig.text(0.035, 0.985, title_text, ha="left", va="top", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.035, 0.945, subtitle_text, ha="left", va="top", fontsize=9, color=TOKENS["muted"])


def normalise_tsiii(nuclei_path: Path, images_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nuclei = pd.read_csv(nuclei_path)
    images = pd.read_csv(images_path)

    nuclei = nuclei.rename(columns={"spots_count": "af488_spots_count"})
    nuclei["time_series"] = "TS III"
    images["time_series"] = "TS III"

    nuclei["rhrex_spots_count"] = np.nan
    nuclei["colocalized_rhrex_spots_count"] = np.nan
    nuclei["af488_has_colocalized_spots_count"] = np.nan

    images = images.rename(
        columns={
            "spot_count": "image_af488_spots_count",
            "fraction_nuclei_with_gt3_spots": "image_fraction_af488_gt3",
        }
    )
    return nuclei, images


def normalise_tsiv(nuclei_path: Path, images_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nuclei = pd.read_csv(nuclei_path)
    images = pd.read_csv(images_path)
    nuclei["time_series"] = "TS IV"
    images["time_series"] = "TS IV"
    return nuclei, images


def field_metrics(nuclei: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "time_series",
        "sample_id",
        "condition",
        "timepoint_hr",
        "replicate",
        "file_name",
    ]
    rows = []
    for key, group in nuclei.groupby(group_cols, sort=False, dropna=False):
        row = dict(zip(group_cols, key))
        af488 = group["af488_spots_count"]
        row.update(
            {
                "n_nuclei_from_nucleus_table": int(len(group)),
                "mean_af488_spots_per_nucleus": float(af488.mean()),
                "median_af488_spots_per_nucleus": float(af488.median()),
                "fraction_nuclei_af488_ge3": float((af488 >= 3).mean()),
                "fraction_nuclei_af488_gt3": float((af488 > 3).mean()),
                "fraction_nuclei_af488_ge5": float((af488 >= 5).mean()),
                "mean_nucleus_area_um2": float(group["nucleus_area_um2"].mean()),
                "median_nucleus_area_um2": float(group["nucleus_area_um2"].median()),
                "mean_nuclear_af488_intensity": float(group["nucleus_mean_af488_intensity"].mean()),
                "mean_nuclear_dapi_intensity": float(group["nucleus_mean_dapi_intensity"].mean()),
            }
        )
        if "rhrex_spots_count" in group.columns:
            rhrex = group["rhrex_spots_count"]
            coloc = group["colocalized_rhrex_spots_count"]
            row.update(
                {
                    "mean_rhrex_spots_per_nucleus": safe_mean(rhrex),
                    "median_rhrex_spots_per_nucleus": safe_median(rhrex),
                    "fraction_nuclei_rhrex_ge1": safe_fraction_ge(rhrex, 1),
                    "mean_colocalized_rhrex_spots_per_nucleus": safe_mean(coloc),
                    "median_colocalized_rhrex_spots_per_nucleus": safe_median(coloc),
                    "fraction_nuclei_colocalized_ge1": safe_fraction_ge(coloc, 1),
                    "fraction_nuclei_colocalized_gt5": safe_fraction_gt(coloc, 5),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def make_field_summary(
    tsiii_nuclei: pd.DataFrame,
    tsiii_images: pd.DataFrame,
    tsiv_nuclei: pd.DataFrame,
    tsiv_images: pd.DataFrame,
) -> pd.DataFrame:
    nuclei_metrics = pd.concat([field_metrics(tsiii_nuclei), field_metrics(tsiv_nuclei)], ignore_index=True)

    base_cols = [
        "time_series",
        "sample_id",
        "condition",
        "timepoint_hr",
        "replicate",
        "file_name",
        "source_path",
        "acquisition_date_folder",
        "acquisition_datetime",
        "nuclei_count",
    ]
    images = pd.concat([tsiii_images[base_cols], tsiv_images[base_cols]], ignore_index=True)
    images = images.rename(columns={"nuclei_count": "n_nuclei_from_image_summary"})

    summary = images.merge(
        nuclei_metrics,
        on=["time_series", "sample_id", "condition", "timepoint_hr", "replicate", "file_name"],
        how="left",
    )
    summary["n_nuclei"] = summary["n_nuclei_from_nucleus_table"].fillna(summary["n_nuclei_from_image_summary"]).fillna(0).astype(int)
    summary["plotted"] = summary["n_nuclei"] > 0
    summary["control_or_treated"] = np.where(summary["condition"].isin(CONTROL_CONDITIONS), "control", "treated")
    summary["timepoint_hr"] = summary["timepoint_hr"].astype(int)
    summary["replicate"] = summary["replicate"].astype(str).str.zfill(2)
    condition_rank = {condition: idx for idx, condition in enumerate(CONDITION_ORDER)}
    summary["_condition_rank"] = summary["condition"].map(condition_rank).fillna(len(CONDITION_ORDER)).astype(int)
    summary = summary.sort_values(["time_series", "timepoint_hr", "_condition_rank", "replicate", "sample_id"])
    return summary.drop(columns=["_condition_rank"]).reset_index(drop=True)


def sem(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) <= 1:
        return np.nan
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def safe_mean(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    return float(values.mean())


def safe_median(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    return float(values.median())


def safe_fraction_ge(values: pd.Series, threshold: float) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    return float((values >= threshold).mean())


def safe_fraction_gt(values: pd.Series, threshold: float) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    return float((values > threshold).mean())


def make_condition_summary(field_summary: pd.DataFrame) -> pd.DataFrame:
    valid = field_summary[field_summary["plotted"]].copy()
    group_cols = ["time_series", "timepoint_hr", "condition", "control_or_treated"]
    rows = []
    for key, group in valid.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, key))
        row.update(
            {
                "n_fields": int(len(group)),
                "total_nuclei": int(group["n_nuclei"].sum()),
                "mean_field_mean_af488_spots_per_nucleus": float(group["mean_af488_spots_per_nucleus"].mean()),
                "sem_field_mean_af488_spots_per_nucleus": sem(group["mean_af488_spots_per_nucleus"]),
                "median_field_mean_af488_spots_per_nucleus": float(group["mean_af488_spots_per_nucleus"].median()),
                "mean_field_fraction_af488_ge3": float(group["fraction_nuclei_af488_ge3"].mean()),
                "mean_field_fraction_af488_ge5": float(group["fraction_nuclei_af488_ge5"].mean()),
                "sem_field_fraction_af488_ge5": sem(group["fraction_nuclei_af488_ge5"]),
                "mean_field_fraction_rhrex_ge1": safe_mean(group["fraction_nuclei_rhrex_ge1"]),
                "mean_field_fraction_colocalized_ge1": safe_mean(group["fraction_nuclei_colocalized_ge1"]),
                "mean_field_mean_rhrex_spots_per_nucleus": safe_mean(group["mean_rhrex_spots_per_nucleus"]),
                "mean_field_mean_colocalized_rhrex_spots_per_nucleus": safe_mean(
                    group["mean_colocalized_rhrex_spots_per_nucleus"]
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def ordered_conditions(values: pd.Series) -> list[str]:
    present = set(values.dropna().astype(str))
    return [condition for condition in CONDITION_ORDER if condition in present]


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=1.0)
    ax.grid(axis="x", visible=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(TOKENS["axis"])
    ax.tick_params(axis="both", labelsize=8)


def plot_field_facets(
    field_summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    subtitle: str,
    output_stem: Path,
    *,
    experiments: list[str],
    timepoints: list[int],
    y_limit: tuple[float, float] | None = None,
) -> None:
    plot_df = field_summary[field_summary["plotted"] & field_summary[metric].notna()].copy()
    rng = np.random.default_rng(20260617)
    fig, axes = plt.subplots(
        len(experiments),
        len(timepoints),
        figsize=(5.1 * len(timepoints), 4.0 * len(experiments)),
        sharey=True,
        squeeze=False,
    )
    add_figure_header(fig, title, subtitle)

    for row_idx, experiment in enumerate(experiments):
        for col_idx, timepoint in enumerate(timepoints):
            ax = axes[row_idx, col_idx]
            subset = plot_df[(plot_df["time_series"] == experiment) & (plot_df["timepoint_hr"] == timepoint)]
            conditions = ordered_conditions(subset["condition"])
            ax.set_title(f"{experiment}, {timepoint} hr", fontsize=10, color=TOKENS["ink"])
            if subset.empty:
                ax.text(
                    0.5,
                    0.5,
                    "not acquired",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color=TOKENS["muted"],
                )
                ax.set_xticks([])
                style_axis(ax)
                continue
            for x_idx, condition in enumerate(conditions):
                values = subset.loc[subset["condition"] == condition, metric].to_numpy(dtype=float)
                x = np.full(values.shape, x_idx, dtype=float) + rng.uniform(-0.13, 0.13, size=values.shape)
                ax.scatter(
                    x,
                    values,
                    s=40,
                    color=CONDITION_COLOURS[condition],
                    edgecolor=CONDITION_EDGES[condition],
                    linewidth=0.8,
                    alpha=0.9,
                    zorder=3,
                )
                mean_value = float(np.nanmean(values))
                ax.plot([x_idx - 0.24, x_idx + 0.24], [mean_value, mean_value], color=TOKENS["ink"], linewidth=1.4, zorder=4)
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, rotation=45, ha="right")
            ax.set_xlim(-0.6, max(len(conditions) - 0.4, 0.6))
            if y_limit is not None:
                ax.set_ylim(*y_limit)
            style_axis(ax)
            if col_idx == 0:
                ax.set_ylabel(ylabel)

    fig.text(
        0.035,
        0.025,
        "Each dot is one CZI image field. Horizontal black ticks show the mean of field replicates in that condition/time panel.",
        fontsize=8,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.995, 0.90))
    fig.savefig(output_stem.with_suffix(".png"), dpi=220)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_tsiv_dual(field_summary: pd.DataFrame, output_stem: Path) -> None:
    metrics = [
        (
            "fraction_nuclei_rhrex_ge1",
            "RhReX+ nuclei fraction",
            "fraction of nuclei with >=1 RhReX focus",
        ),
        (
            "fraction_nuclei_colocalized_ge1",
            "Co-localised nuclei fraction",
            "fraction of nuclei with >=1 RhReX focus overlapping AF488",
        ),
        (
            "mean_colocalized_rhrex_spots_per_nucleus",
            "Mean co-localised foci",
            "co-localised RhReX foci per nucleus",
        ),
    ]
    plot_df = field_summary[(field_summary["time_series"] == "TS IV") & field_summary["plotted"]].copy()
    rng = np.random.default_rng(20260617)
    fig, axes = plt.subplots(len(metrics), len(TIMEPOINT_ORDER), figsize=(15.5, 10.5), squeeze=False)
    add_figure_header(
        fig,
        "TS IV dual-channel response by field replicate",
        "Dots are CZI fields; RhReX is the putative gH2AX channel from metadata channel C0. Co-localisation uses the earlier 1-pixel AF488-overlap rule within DAPI nuclei.",
    )
    for row_idx, (metric, row_title, ylabel) in enumerate(metrics):
        y_values = plot_df[metric].dropna()
        if y_values.empty:
            y_limit = None
        elif "fraction" in metric:
            y_limit = (0, min(1.02, max(0.35, float(y_values.max()) * 1.18)))
        else:
            y_limit = (0, max(0.2, float(y_values.max()) * 1.18))
        for col_idx, timepoint in enumerate(TIMEPOINT_ORDER):
            ax = axes[row_idx, col_idx]
            subset = plot_df[(plot_df["timepoint_hr"] == timepoint) & plot_df[metric].notna()]
            conditions = ordered_conditions(subset["condition"])
            ax.set_title(f"{row_title}, {timepoint} hr", fontsize=10, color=TOKENS["ink"])
            for x_idx, condition in enumerate(conditions):
                values = subset.loc[subset["condition"] == condition, metric].to_numpy(dtype=float)
                x = np.full(values.shape, x_idx, dtype=float) + rng.uniform(-0.13, 0.13, size=values.shape)
                ax.scatter(
                    x,
                    values,
                    s=40,
                    color=CONDITION_COLOURS[condition],
                    edgecolor=CONDITION_EDGES[condition],
                    linewidth=0.8,
                    alpha=0.9,
                    zorder=3,
                )
                mean_value = float(np.nanmean(values))
                ax.plot([x_idx - 0.24, x_idx + 0.24], [mean_value, mean_value], color=TOKENS["ink"], linewidth=1.4, zorder=4)
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, rotation=45, ha="right")
            ax.set_xlim(-0.6, max(len(conditions) - 0.4, 0.6))
            if y_limit is not None:
                ax.set_ylim(*y_limit)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            style_axis(ax)
    fig.text(
        0.035,
        0.025,
        "Each dot is one CZI image field. Horizontal black ticks show the mean of field replicates in that condition/time panel.",
        fontsize=8,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.995, 0.90))
    fig.savefig(output_stem.with_suffix(".png"), dpi=220)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def write_notes(field_summary: pd.DataFrame, condition_summary: pd.DataFrame, excluded: pd.DataFrame, output: Path) -> None:
    valid = field_summary[field_summary["plotted"]].copy()
    lines = [
        "Field-level summary for biological interpretation",
        "",
        "Observation unit:",
        "- One replicate point is one CZI image field.",
        "- Per-nucleus measurements were averaged or converted to fractions within each field before plotting.",
        "",
        "Scope:",
        f"- TS III fields in source image summary: {int((field_summary['time_series'] == 'TS III').sum())}.",
        f"- TS IV fields in source image summary: {int((field_summary['time_series'] == 'TS IV').sum())}.",
        f"- Fields plotted after excluding zero-nucleus fields: {len(valid)}.",
    ]
    if not excluded.empty:
        lines.extend(
            [
                "",
                "Excluded from field plots because no nuclei were segmented:",
            ]
        )
        for _, row in excluded.iterrows():
            lines.append(f"- {row['time_series']} {row['sample_id']} ({row['file_name']})")
    lines.extend(
        [
            "",
            "Biological readout caveats:",
            "- AF488 can be compared across TS III and TS IV at the analysis level, but the experiments were acquired on different dates and TS IV includes a 24 hr set.",
            "- RhReX and co-localisation summaries are TS IV only; TS III was not analysed for RhReX/co-localisation.",
            "- RhReX is labelled as putative gH2AX by experimental context; the CZI metadata identifies it as RhReX-T1/Rhodamine Red-X.",
            "- The staining and segmentation QC caveats from earlier steps still apply, so interpret field means as screening-level summaries rather than final biological effect sizes.",
            "",
            "Highest AF488 field-mean conditions by experiment and time:",
        ]
    )
    for (experiment, timepoint), group in condition_summary.groupby(["time_series", "timepoint_hr"], sort=False):
        ranked = group.sort_values("mean_field_mean_af488_spots_per_nucleus", ascending=False).head(3)
        formatted = ", ".join(
            f"{row.condition} {row.mean_field_mean_af488_spots_per_nucleus:.2f}"
            for row in ranked.itertuples(index=False)
        )
        lines.append(f"- {experiment} {timepoint} hr: {formatted}")

    tsiv_dual = condition_summary[condition_summary["time_series"] == "TS IV"].copy()
    if not tsiv_dual.empty:
        lines.extend(["", "Highest TS IV co-localised field fractions by time:"])
        for timepoint, group in tsiv_dual.groupby("timepoint_hr", sort=False):
            ranked = group.sort_values("mean_field_fraction_colocalized_ge1", ascending=False).head(3)
            formatted = ", ".join(
                f"{row.condition} {row.mean_field_fraction_colocalized_ge1:.3f}"
                for row in ranked.itertuples(index=False)
            )
            lines.append(f"- TS IV {timepoint} hr: {formatted}")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tsiii_nuclei, tsiii_images = normalise_tsiii(args.tsiii_nuclei, args.tsiii_images)
    tsiv_nuclei, tsiv_images = normalise_tsiv(args.tsiv_nuclei, args.tsiv_images)
    field_summary = make_field_summary(tsiii_nuclei, tsiii_images, tsiv_nuclei, tsiv_images)
    condition_summary = make_condition_summary(field_summary)
    excluded = field_summary[~field_summary["plotted"]].copy()

    field_summary.to_csv(args.output_dir / "field_replicate_summary_TSIII_TSIV.csv", index=False)
    condition_summary.to_csv(args.output_dir / "condition_time_field_summary_TSIII_TSIV.csv", index=False)
    excluded.to_csv(args.output_dir / "excluded_fields_no_nuclei.csv", index=False)

    plot_field_facets(
        field_summary,
        metric="mean_af488_spots_per_nucleus",
        ylabel="Mean AF488 spots per nucleus",
        title="AF488 foci burden by field replicate across TS III and TS IV",
        subtitle="One dot is one CZI field. Values are field-level means of AF488 spots counted inside DAPI-segmented nuclei; black ticks are condition/time field means.",
        output_stem=args.output_dir / "AF488_mean_spots_per_nucleus_field_replicates",
        experiments=EXPERIMENT_ORDER,
        timepoints=TIMEPOINT_ORDER,
    )
    plot_field_facets(
        field_summary,
        metric="fraction_nuclei_af488_ge5",
        ylabel="Fraction nuclei with >=5 AF488 spots",
        title="AF488-positive nuclei by field replicate across TS III and TS IV",
        subtitle="The >=5 nucleus threshold is shown for cross-experiment comparability; TS III's earlier empirical AF488 threshold was lower, so this is a stricter TS III readout.",
        output_stem=args.output_dir / "AF488_fraction_nuclei_ge5_field_replicates",
        experiments=EXPERIMENT_ORDER,
        timepoints=TIMEPOINT_ORDER,
        y_limit=(0, 1.02),
    )
    plot_tsiv_dual(field_summary, args.output_dir / "TSIV_RhReX_colocalisation_field_replicates")
    write_notes(field_summary, condition_summary, excluded, args.output_dir / "biological_interpretation_notes.txt")

    print(f"Wrote field-level summary outputs to {args.output_dir}")
    print(f"Field rows: {len(field_summary)}")
    print(f"Fields plotted: {int(field_summary['plotted'].sum())}")
    print(f"Excluded zero-nucleus fields: {len(excluded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
