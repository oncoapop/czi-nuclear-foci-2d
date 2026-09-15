#!/usr/bin/env python3
"""Summarise all CZI dual-channel experiments at field and condition level."""

from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENTS = {
    "TS III": {
        "folder": Path("output/segmentation/time_series_III_dual_channel_coloc"),
        "note": "quarter dilution; named conditions; 2 hr and 6 hr",
        "known_controls": {"Aqueous", "DMSO", "NAH2PO4"},
    },
    "TS IV": {
        "folder": Path("output/segmentation/time_series_IV_dual_channel_coloc"),
        "note": "weird controls; named conditions; 2 hr, 6 hr, and 24 hr",
        "known_controls": {"Aqueous", "DMSO", "NAH2PO4"},
    },
    "TS V": {
        "folder": Path("output/segmentation/time_series_V_dual_channel_coloc"),
        "note": "24 hr and 48 hr; A-H mapped from user-provided TS V setup image",
        "known_controls": {"NAH2PO4", "Media", "DMSO"},
    },
}

NAMED_CONDITION_ORDER = [
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
TIMEPOINT_ORDER = [2, 6, 24, 48]
TSV_MAPPING_PATH = Path("time_series_V_condition_mapping.csv")

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

PALETTE = [
    "#464C55",
    "#C5CAD3",
    "#7A828F",
    "#A3BEFA",
    "#F0986E",
    "#F390CA",
    "#A3D576",
    "#FFE15B",
]
EDGES = [
    "#1F2430",
    "#7A828F",
    "#464C55",
    "#2E4780",
    "#804126",
    "#8A3A6F",
    "#386411",
    "#736422",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/segmentation/all_experiments_dual_channel_summary"))
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


def load_tsv_mapping() -> pd.DataFrame:
    mapping = pd.read_csv(TSV_MAPPING_PATH, dtype={"condition_code": str})
    required = {
        "condition_code",
        "well_number",
        "condition",
        "concentration",
        "condition_type",
        "is_vehicle_or_control",
        "source_note",
    }
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"{TSV_MAPPING_PATH} is missing required columns: {sorted(missing)}")
    mapping["condition_code"] = mapping["condition_code"].astype(str)
    mapping["condition"] = mapping["condition"].astype(str)
    return mapping


def apply_tsv_mapping(frame: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["condition_code"] = frame["condition"].astype(str)
    mapped = frame.merge(mapping, on="condition_code", how="left", suffixes=("_original", "_mapped"))
    missing = sorted(mapped.loc[mapped["condition_mapped"].isna(), "condition_code"].dropna().unique())
    if missing:
        raise ValueError(f"TS V conditions missing from {TSV_MAPPING_PATH}: {missing}")
    mapped["condition"] = mapped["condition_mapped"]
    mapped["condition_concentration"] = mapped["concentration"]
    mapped["condition_mapping_source"] = mapped["source_note"]
    mapped = mapped.drop(columns=["condition_original", "condition_mapped", "concentration", "source_note"])
    return mapped


def add_unmapped_condition_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["condition_code"] = ""
    frame["well_number"] = pd.NA
    frame["condition_concentration"] = ""
    frame["condition_type"] = ""
    frame["is_vehicle_or_control"] = pd.NA
    frame["condition_mapping_source"] = ""
    return frame


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    image_frames = []
    nucleus_frames = []
    tsv_mapping = load_tsv_mapping()
    for experiment, config in EXPERIMENTS.items():
        folder = config["folder"]
        images = pd.read_csv(folder / "image_summary_dual_channel.csv")
        nuclei = pd.read_csv(folder / "nucleus_dual_channel_measurements.csv")
        if experiment == "TS V":
            images = apply_tsv_mapping(images, tsv_mapping)
            nuclei = apply_tsv_mapping(nuclei, tsv_mapping)
        else:
            images = add_unmapped_condition_columns(images)
            nuclei = add_unmapped_condition_columns(nuclei)
        for frame in (images, nuclei):
            frame["experiment"] = experiment
            frame["experiment_note"] = config["note"]
            frame["condition_is_coded"] = frame["condition_code"].astype(str).str.fullmatch(r"[A-H]").fillna(False)
            frame["known_control"] = frame["condition"].isin(config["known_controls"])
            frame["timepoint_hr"] = frame["timepoint_hr"].astype(int)
            frame["replicate"] = frame["replicate"].astype(str).str.zfill(2)
        image_frames.append(images)
        nucleus_frames.append(nuclei)
    return pd.concat(image_frames, ignore_index=True), pd.concat(nucleus_frames, ignore_index=True)


def safe_mean(values: pd.Series) -> float:
    values = values.dropna()
    return float(values.mean()) if len(values) else np.nan


def safe_sem(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) <= 1:
        return np.nan
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def condition_order_for(experiment: str, conditions: pd.Series) -> list[str]:
    present = set(conditions.dropna().astype(str))
    ordered = [condition for condition in NAMED_CONDITION_ORDER if condition in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def make_field_summary(images: pd.DataFrame, nuclei: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["experiment", "sample_id", "condition", "timepoint_hr", "replicate", "file_name"]
    rows = []
    for key, group in nuclei.groupby(group_cols, sort=False, dropna=False):
        row = dict(zip(group_cols, key))
        row.update(
            {
                "n_nuclei_from_nucleus_table": int(len(group)),
                "mean_af488_spots_per_nucleus": safe_mean(group["af488_spots_count"]),
                "median_af488_spots_per_nucleus": float(group["af488_spots_count"].median()),
                "fraction_nuclei_af488_ge3": float((group["af488_spots_count"] >= 3).mean()),
                "fraction_nuclei_af488_ge5": float((group["af488_spots_count"] >= 5).mean()),
                "mean_rhrex_spots_per_nucleus": safe_mean(group["rhrex_spots_count"]),
                "median_rhrex_spots_per_nucleus": float(group["rhrex_spots_count"].median()),
                "fraction_nuclei_rhrex_ge1": float((group["rhrex_spots_count"] >= 1).mean()),
                "mean_colocalized_rhrex_spots_per_nucleus": safe_mean(group["colocalized_rhrex_spots_count"]),
                "median_colocalized_rhrex_spots_per_nucleus": float(group["colocalized_rhrex_spots_count"].median()),
                "fraction_nuclei_colocalized_ge1": float((group["colocalized_rhrex_spots_count"] >= 1).mean()),
                "fraction_nuclei_colocalized_gt5": float((group["colocalized_rhrex_spots_count"] > 5).mean()),
                "mean_nucleus_area_um2": safe_mean(group["nucleus_area_um2"]),
                "mean_nuclear_dapi_intensity": safe_mean(group["nucleus_mean_dapi_intensity"]),
                "mean_nuclear_af488_intensity": safe_mean(group["nucleus_mean_af488_intensity"]),
                "mean_nuclear_rhrex_intensity": safe_mean(group["nucleus_mean_rhrex_intensity"]),
            }
        )
        rows.append(row)
    field = pd.DataFrame(rows)
    image_cols = [
        "experiment",
        "sample_id",
        "condition",
        "condition_code",
        "well_number",
        "condition_concentration",
        "condition_type",
        "is_vehicle_or_control",
        "condition_mapping_source",
        "condition_is_coded",
        "known_control",
        "timepoint_hr",
        "replicate",
        "file_name",
        "source_path",
        "acquisition_date_folder",
        "acquisition_datetime",
        "channel_0_metadata",
        "channel_1_metadata",
        "channel_2_metadata",
        "nuclei_count",
        "af488_threshold",
        "rhrex_threshold",
    ]
    merged = images[image_cols].merge(field, on=group_cols, how="left")
    merged = merged.rename(columns={"nuclei_count": "n_nuclei_from_image_summary"})
    merged["n_nuclei"] = merged["n_nuclei_from_nucleus_table"].fillna(merged["n_nuclei_from_image_summary"]).fillna(0).astype(int)
    merged["plotted"] = merged["n_nuclei"] > 0
    return merged.sort_values(["experiment", "timepoint_hr", "condition", "replicate", "sample_id"]).reset_index(drop=True)


def make_condition_summary(field: pd.DataFrame) -> pd.DataFrame:
    valid = field[field["plotted"]].copy()
    rows = []
    for (experiment, timepoint, condition), group in valid.groupby(["experiment", "timepoint_hr", "condition"], sort=False):
        rows.append(
            {
                "experiment": experiment,
                "timepoint_hr": int(timepoint),
                "condition": condition,
                "condition_is_coded": bool(group["condition_is_coded"].iloc[0]),
                "n_fields": int(len(group)),
                "total_nuclei": int(group["n_nuclei"].sum()),
                "mean_field_mean_af488_spots_per_nucleus": safe_mean(group["mean_af488_spots_per_nucleus"]),
                "sem_field_mean_af488_spots_per_nucleus": safe_sem(group["mean_af488_spots_per_nucleus"]),
                "mean_field_fraction_af488_ge5": safe_mean(group["fraction_nuclei_af488_ge5"]),
                "sem_field_fraction_af488_ge5": safe_sem(group["fraction_nuclei_af488_ge5"]),
                "mean_field_mean_rhrex_spots_per_nucleus": safe_mean(group["mean_rhrex_spots_per_nucleus"]),
                "sem_field_mean_rhrex_spots_per_nucleus": safe_sem(group["mean_rhrex_spots_per_nucleus"]),
                "mean_field_fraction_rhrex_ge1": safe_mean(group["fraction_nuclei_rhrex_ge1"]),
                "mean_field_mean_colocalized_rhrex_spots_per_nucleus": safe_mean(
                    group["mean_colocalized_rhrex_spots_per_nucleus"]
                ),
                "sem_field_mean_colocalized_rhrex_spots_per_nucleus": safe_sem(
                    group["mean_colocalized_rhrex_spots_per_nucleus"]
                ),
                "mean_field_fraction_colocalized_ge1": safe_mean(group["fraction_nuclei_colocalized_ge1"]),
                "sem_field_fraction_colocalized_ge1": safe_sem(group["fraction_nuclei_colocalized_ge1"]),
            }
        )
    return pd.DataFrame(rows)


def threshold_performance(nuclei: pd.DataFrame, metric: str) -> pd.DataFrame:
    named = nuclei[nuclei["known_control"].notna()].copy()
    controls = named["known_control"].astype(bool)
    treated = ~controls
    rows = []
    max_count = int(named[metric].max()) if len(named) else 0
    for threshold in range(max_count + 2):
        positive = named[metric] >= threshold
        tp = int((positive & treated).sum())
        fp = int((positive & controls).sum())
        tn = int((~positive & controls).sum())
        fn = int((~positive & treated).sum())
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        rows.append(
            {
                "scope": "all mapped experiments; controls are Aqueous, DMSO, NAH2PO4, and Media where present",
                "metric": metric,
                "positive_if_count_ge": threshold,
                "true_positive_treated_nuclei": tp,
                "false_positive_control_nuclei": fp,
                "true_negative_control_nuclei": tn,
                "false_negative_treated_nuclei": fn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "youden_j": sensitivity + specificity - 1.0,
                "control_positive_fraction": fp / int(controls.sum()) if controls.sum() else 0.0,
                "treated_positive_fraction": tp / int(treated.sum()) if treated.sum() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(TOKENS["axis"])
    ax.tick_params(axis="both", labelsize=8)


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.035, 0.985, textwrap.fill(title, width=100), ha="left", va="top", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.035, 0.945, textwrap.fill(subtitle, width=135), ha="left", va="top", fontsize=9, color=TOKENS["muted"])


def plot_metric(field: pd.DataFrame, metric: str, ylabel: str, title: str, subtitle: str, output_stem: Path) -> None:
    plot_df = field[field["plotted"] & field[metric].notna()].copy()
    experiments = ["TS III", "TS IV", "TS V"]
    timepoints = TIMEPOINT_ORDER
    rng = np.random.default_rng(20260618)
    fig, axes = plt.subplots(len(experiments), len(timepoints), figsize=(5.0 * len(timepoints), 3.8 * len(experiments)), sharey=True, squeeze=False)
    add_header(fig, title, subtitle)
    metric_max = plot_df[metric].max()
    if pd.notna(metric_max) and "fraction" in metric:
        y_limit = (0, min(1.02, max(0.35, float(metric_max) * 1.18)))
    elif pd.notna(metric_max):
        y_limit = (0, max(0.2, float(metric_max) * 1.18))
    else:
        y_limit = None

    for row_idx, experiment in enumerate(experiments):
        for col_idx, timepoint in enumerate(timepoints):
            ax = axes[row_idx, col_idx]
            subset = plot_df[(plot_df["experiment"] == experiment) & (plot_df["timepoint_hr"] == timepoint)]
            ax.set_title(f"{experiment}, {timepoint} hr", fontsize=10)
            if subset.empty:
                ax.text(0.5, 0.5, "not acquired", ha="center", va="center", transform=ax.transAxes, color=TOKENS["muted"])
                ax.set_xticks([])
                style_axis(ax)
                continue
            conditions = condition_order_for(experiment, subset["condition"])
            for idx, condition in enumerate(conditions):
                values = subset.loc[subset["condition"] == condition, metric].to_numpy(dtype=float)
                x = np.full(values.shape, idx, dtype=float) + rng.uniform(-0.13, 0.13, size=values.shape)
                colour = PALETTE[idx % len(PALETTE)]
                edge = EDGES[idx % len(EDGES)]
                ax.scatter(x, values, s=40, color=colour, edgecolor=edge, linewidth=0.8, alpha=0.9, zorder=3)
                ax.plot([idx - 0.24, idx + 0.24], [np.nanmean(values), np.nanmean(values)], color=TOKENS["ink"], linewidth=1.4)
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, rotation=45, ha="right")
            ax.set_xlim(-0.6, max(len(conditions) - 0.4, 0.6))
            if y_limit:
                ax.set_ylim(*y_limit)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            style_axis(ax)
    fig.text(0.035, 0.025, "Each dot is one CZI image field. Horizontal black ticks are condition/time field means.", fontsize=8, color=TOKENS["muted"])
    fig.tight_layout(rect=(0.02, 0.05, 0.995, 0.90))
    fig.savefig(output_stem.with_suffix(".png"), dpi=220)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def top_rows(summary: pd.DataFrame, metric: str, n: int = 3) -> list[str]:
    lines = []
    for (experiment, timepoint), group in summary.groupby(["experiment", "timepoint_hr"], sort=False):
        ranked = group.sort_values(metric, ascending=False).head(n)
        formatted = ", ".join(f"{row.condition} {getattr(row, metric):.3f}" for row in ranked.itertuples(index=False))
        lines.append(f"- {experiment} {timepoint} hr: {formatted}")
    return lines


def write_report(
    images: pd.DataFrame,
    nuclei: pd.DataFrame,
    field: pd.DataFrame,
    condition_summary: pd.DataFrame,
    best_thresholds: pd.DataFrame,
    output: Path,
) -> None:
    lines = [
        "# All CZI Experiments Dual-Channel Summary",
        "",
        "## Scope",
        "",
        f"- CZI image fields analysed: {len(images)}.",
        f"- Fields with at least one segmented nucleus: {int(field['plotted'].sum())}.",
        f"- Segmented nuclei: {len(nuclei)}.",
        f"- AF488 spots: {int(images['af488_spots_count'].sum())}.",
        f"- RhReX putative gH2AX foci: {int(images['rhrex_spots_count'].sum())}.",
        f"- Co-localised RhReX foci: {int(images['colocalized_rhrex_spots_count'].sum())}.",
        "",
        "## Channel Metadata",
        "",
    ]
    channels = images[["experiment", "channel_0_metadata", "channel_1_metadata", "channel_2_metadata"]].drop_duplicates()
    for row in channels.itertuples(index=False):
        lines.append(f"- {row.experiment}: {row.channel_0_metadata}; {row.channel_1_metadata}; {row.channel_2_metadata}.")
    lines.extend(
        [
            "",
            "## Analysis Grain",
            "",
            "- Replicate unit for summary plots: one CZI image field.",
            "- Nuclei: segmented from DAPI-T3.",
            "- AF488 and RhReX spots: segmented independently inside DAPI nuclear masks.",
            "- Co-localisation: RhReX focus overlapping an AF488 spot after 1 pixel dilation, inside a DAPI nucleus.",
            "- TS V conditions A-H were mapped to treatment names from the user-provided TS V setup image; original A-H labels are retained as condition_code.",
            "",
            "## Top Field-Level AF488 Burden",
            "",
        ]
    )
    lines.extend(top_rows(condition_summary, "mean_field_mean_af488_spots_per_nucleus"))
    lines.extend(["", "## Top Field-Level RhReX Burden", ""])
    lines.extend(top_rows(condition_summary, "mean_field_mean_rhrex_spots_per_nucleus"))
    lines.extend(["", "## Top Field-Level Co-localised Fraction", ""])
    lines.extend(top_rows(condition_summary, "mean_field_fraction_colocalized_ge1"))
    lines.extend(["", "## Mapped-Condition Threshold Evidence", ""])
    for row in best_thresholds.itertuples(index=False):
        lines.append(
            f"- {row.metric}: best simple cut-off >= {int(row.positive_if_count_ge)}; "
            f"sensitivity {row.sensitivity:.3f}; specificity {row.specificity:.3f}; "
            f"control positive fraction {row.control_positive_fraction:.3f}; treated positive fraction {row.treated_positive_fraction:.3f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation Caveats",
            "",
            "- RhReX is labelled putative gH2AX by experimental context; CZI metadata names the channel RhReX-T1/Rhodamine Red-X.",
            "- TS V treatment identities are based on the user-provided experimental setup image; original A-H codes are retained for traceability.",
            "- These are screening-level image-analysis summaries. Poor DAPI staining and imperfect nuclei segmentation remain visible confounders.",
            "- Cross-experiment comparisons are vulnerable to batch/date effects and should be interpreted primarily within each time series unless normalised with a confirmed control strategy.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images, nuclei = load_tables()
    field = make_field_summary(images, nuclei)
    condition_summary = make_condition_summary(field)

    images.to_csv(args.output_dir / "combined_image_summary_dual_channel.csv", index=False)
    nuclei.to_csv(args.output_dir / "combined_nucleus_dual_channel_measurements.csv", index=False)
    field.to_csv(args.output_dir / "field_replicate_summary_all_experiments.csv", index=False)
    condition_summary.to_csv(args.output_dir / "condition_time_field_summary_all_experiments.csv", index=False)

    best_rows = []
    threshold_frames = []
    for metric in ["af488_spots_count", "rhrex_spots_count", "colocalized_rhrex_spots_count"]:
        table = threshold_performance(nuclei, metric)
        table.to_csv(args.output_dir / f"threshold_named_controls_vs_treated_{metric}.csv", index=False)
        threshold_frames.append(table)
        best_rows.append(table.sort_values(["youden_j", "specificity", "sensitivity"], ascending=False).iloc[0].to_dict())
    thresholds = pd.concat(threshold_frames, ignore_index=True)
    thresholds.to_csv(args.output_dir / "threshold_named_controls_vs_treated_all_metrics.csv", index=False)
    best_thresholds = pd.DataFrame(best_rows)
    best_thresholds.to_csv(args.output_dir / "best_thresholds_named_controls_vs_treated.csv", index=False)

    plot_metric(
        field,
        "mean_af488_spots_per_nucleus",
        "Mean AF488 spots per nucleus",
        "AF488 foci burden across all CZI experiments",
        "Dots are CZI fields. TS V A-H labels were mapped to treatment names from the supplied setup image.",
        args.output_dir / "all_experiments_AF488_mean_spots_per_nucleus_field_replicates",
    )
    plot_metric(
        field,
        "fraction_nuclei_af488_ge5",
        "Fraction nuclei with >=5 AF488 spots",
        "AF488-positive nuclei across all CZI experiments",
        "The >=5 threshold is shown consistently; threshold optimisation uses mapped controls from all experiments.",
        args.output_dir / "all_experiments_AF488_fraction_ge5_field_replicates",
    )
    plot_metric(
        field,
        "mean_rhrex_spots_per_nucleus",
        "Mean RhReX foci per nucleus",
        "RhReX putative gH2AX burden across all CZI experiments",
        "RhReX-T1/Rhodamine Red-X is treated as putative gH2AX by experimental context, not by CZI metadata.",
        args.output_dir / "all_experiments_RhReX_mean_spots_per_nucleus_field_replicates",
    )
    plot_metric(
        field,
        "fraction_nuclei_colocalized_ge1",
        "Fraction nuclei with >=1 co-localised focus",
        "Co-localised RhReX/AF488 nuclei across all CZI experiments",
        "Co-localisation is a RhReX focus intersecting an AF488 spot after 1 pixel dilation within DAPI nuclei.",
        args.output_dir / "all_experiments_colocalized_fraction_ge1_field_replicates",
    )

    write_report(images, nuclei, field, condition_summary, best_thresholds, args.output_dir / "all_experiments_summary_report.md")

    print(f"Wrote all-experiment summaries to {args.output_dir}")
    print(f"Images: {len(images)}")
    print(f"Nuclei: {len(nuclei)}")
    print(f"Fields plotted: {int(field['plotted'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
