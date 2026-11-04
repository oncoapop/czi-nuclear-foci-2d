#!/usr/bin/env python3
"""Publication 53BP1 figures on the fold-change endpoint.

Replaces the absolute-value figures in `make_publication_53bp1.py`. Same input
data, same panel layout, different y-axis.

Why: the 2026-09-15 audit measured what survives a change to the detection
threshold (`threshold_mad_multiplier`). Absolute foci counts do not - moving the
multiplier 3 -> 5 changes total 53BP1 counts 1.32x -> 0.75x, so "4 foci per
nucleus" is as much a statement about the parameter as about the biology.
Ratios to within-batch controls do survive: median spread 1.23x across
multiplier 3-5, direction stable in 30/35 condition-blocks.

Batch is experiment x timepoint, not experiment alone. TS III control levels
fall 1.56x between 2 h and 6 h in this dataset (2.59 -> 1.66 mean foci per
nucleus) on a single acquisition date, so each timepoint gets its own baseline.

Unit of analysis is the imaging field. Nuclei within a field are not
independent; treating them as replicates inflates significance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path(
    "/Users/dyap/Documents/Tasks workflow/czi-analysis/output/publication_53BP1_v1/data"
)
DEFAULT_OUT = ROOT / "output" / "publication_53BP1_foldchange_v1"

EXPERIMENTS = ["TSIII", "TSIV", "TSV"]
LAYOUT = {"TSIII": [2, 6], "TSIV": [2, 6, 24], "TSV": [24, 48]}
CONDITION_ORDER = ["Aqueous", "DMSO", "NAH2PO4", "Media",
                   "CX-5461", "Cisplatin", "Etoposide", "PDS", "Palbociclib"]
CONTROLS = {"Aqueous", "DMSO", "NAH2PO4", "Media"}
POSITIVE_THRESHOLD = 5
TSV_MAP = {"A": "CX-5461", "B": "Cisplatin", "C": "Etoposide", "D": "Palbociclib",
           "E": "NAH2PO4", "F": "Media", "G": "DMSO", "H": "PDS"}

TREATMENT_COLOUR = "#087E8B"
CONTROL_COLOUR = "#747474"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load(data: Path, name: str) -> pd.DataFrame:
    frames = []
    for experiment in EXPERIMENTS:
        frame = pd.read_csv(data / experiment / name)
        frame["experiment"] = experiment
        if experiment == "TSV":
            frame["condition"] = frame["condition"].map(TSV_MAP)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def field_table(nuclei: pd.DataFrame, images: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per imaging field. Fields with zero accepted nuclei are dropped."""
    valid = images.loc[images["nuclei_count"] > 0]
    excluded = images.loc[images["nuclei_count"] <= 0].copy()
    keys = ["experiment", "sample_id", "file_name", "condition", "timepoint_hr", "replicate"]
    fields = nuclei.groupby(keys, as_index=False).agg(
        nuclei_count=("nucleus_label", "size"),
        nuclei_gt5_53BP1=("53BP1_count", lambda x: int((x > POSITIVE_THRESHOLD).sum())),
        mean_53BP1_foci_per_nucleus=("53BP1_count", "mean"),
        median_53BP1_foci_per_nucleus=("53BP1_count", "median"),
    )
    fields = fields.merge(valid[["experiment", "sample_id"]], on=["experiment", "sample_id"])
    fields["percent_nuclei_gt5_53BP1"] = (
        100.0 * fields["nuclei_gt5_53BP1"] / fields["nuclei_count"]
    )
    fields["is_control"] = fields["condition"].isin(CONTROLS)
    return fields, excluded


def add_fold_changes(fields: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Divide every field by the mean of its own batch's control fields.

    Controls are divided by the same denominator, so their scatter around 1.0 is
    the empirical noise floor the treatment effects have to clear.
    """
    out = fields.copy()
    for metric in metrics:
        baseline = (
            out.loc[out["is_control"]]
            .groupby(["experiment", "timepoint_hr"])[metric]
            .mean()
            .rename(f"{metric}__control_mean")
        )
        out = out.merge(baseline, on=["experiment", "timepoint_hr"], how="left")
        out[f"{metric}__fold"] = out[metric] / out[f"{metric}__control_mean"]
    return out


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return out


def effects(fields: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Welch t-test, each treatment against the pooled controls of its own batch."""
    rows = []
    for (experiment, timepoint), block in fields.groupby(["experiment", "timepoint_hr"]):
        control = block.loc[block["is_control"], metric]
        if len(control) < 2:
            continue
        for condition, group in block.loc[~block["is_control"]].groupby("condition"):
            if len(group) < 2:
                continue
            test = stats.ttest_ind(group[metric], control, equal_var=False)
            pooled = np.sqrt((group[metric].var(ddof=1) + control.var(ddof=1)) / 2)
            rows.append({
                "experiment": experiment, "timepoint_hr": timepoint, "condition": condition,
                "n_treated_fields": len(group), "n_control_fields": len(control),
                "control_mean": control.mean(), "treated_mean": group[metric].mean(),
                "fold_change": group[metric].mean() / control.mean() if control.mean() else np.nan,
                "cohens_d": (group[metric].mean() - control.mean()) / pooled if pooled else np.nan,
                "p_welch": test.pvalue,
            })
    out = pd.DataFrame(rows)
    out["q_bh"] = benjamini_hochberg(out["p_welch"].to_numpy())
    return out


def stars(q: float) -> str:
    if not np.isfinite(q):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "ns"


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def plot_fold_change(fields: pd.DataFrame, eff: pd.DataFrame, metric: str,
                     *, title: str, ylabel: str, baseline_fmt: str,
                     caveat: str, stamp: str, out_dir: Path, stem: str) -> dict:
    fold = f"{metric}__fold"
    baseline_col = f"{metric}__control_mean"
    rng = np.random.default_rng(20260717)
    key = eff.set_index(["experiment", "timepoint_hr", "condition"])

    finite = fields.loc[np.isfinite(fields[fold]) & (fields[fold] > 0), fold]
    clipped = int((fields[fold] <= 0).sum())
    ymin = min(0.4, float(finite.min()) * (0.55 if clipped else 0.85))
    ymax = max(2.5, float(finite.max()) * 2.1)   # headroom for the significance row
    floor = ymin * 1.15                          # zero-valued fields sit here, inside the axis

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 12.5), sharey=True)
    for row, (experiment, times) in enumerate(LAYOUT.items()):
        for col in range(3):
            ax = axes[row, col]
            if col >= len(times):
                ax.axis("off")
                continue
            timepoint = times[col]
            sub = fields[(fields["experiment"] == experiment)
                         & (fields["timepoint_hr"] == timepoint)]
            present = [c for c in CONDITION_ORDER if c in set(sub["condition"])]

            ax.axhline(1.0, color="#B00020", linewidth=1.1, linestyle="--", zorder=2)
            for x, condition in enumerate(present):
                rows_ = sub[sub["condition"] == condition]
                values = rows_[fold].to_numpy(float)
                at_floor = values <= 0
                values = np.where(at_floor, floor, values)
                colour = CONTROL_COLOUR if condition in CONTROLS else TREATMENT_COLOUR
                jitter = rng.uniform(-0.10, 0.10, len(values))
                ax.scatter(np.full(len(values), x) + jitter, values, s=34,
                           facecolor=colour, edgecolor="white", linewidth=0.6,
                           marker="o", zorder=4)
                if at_floor.any():
                    ax.scatter(np.full(at_floor.sum(), x) + jitter[at_floor],
                               np.full(at_floor.sum(), floor), s=46, facecolor="none",
                               edgecolor="#B00020", linewidth=1.1, marker="v", zorder=5)
                median = float(np.median(values))
                q1, q3 = np.quantile(values, [0.25, 0.75])
                ax.vlines(x, q1, q3, color="black", linewidth=1.3, zorder=6)
                ax.hlines(median, x - 0.20, x + 0.20, color="black", linewidth=2.2, zorder=7)

                if condition not in CONTROLS:
                    try:
                        q = float(key.loc[(experiment, timepoint, condition), "q_bh"])
                    except KeyError:
                        continue
                    ax.text(x, ymax / 1.45, stars(q), ha="center", va="center",
                            fontsize=9, color="#333333")

            baseline = float(sub[baseline_col].iloc[0])
            ax.set_title(f"{experiment.replace('TS', 'TS ')} - {timepoint} h",
                         loc="left", fontweight="bold")
            ax.set_title(baseline_fmt.format(baseline=baseline), loc="right",
                         fontsize=8.5, color="#555555")
            ax.set_yscale("log", base=2)
            ax.set_ylim(ymin, ymax)
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(
                    lambda v, _: f"{v:g}" if v >= 1 else f"{v:.2f}".rstrip("0")))
            ax.set_xticks(range(len(present)))
            ax.set_xticklabels(present, rotation=42, ha="right")
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(ylabel)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.998)
    footer = (
        "Each point is one imaging field, divided by the mean of the control fields of its own "
        "batch (experiment x timepoint). Grey = control, teal = treatment; bar and whisker are "
        "median and IQR across fields.\n"
        "Dashed line at 1.0 is the control baseline: grey scatter around it is the empirical "
        "noise floor a treatment effect has to clear. Absolute control level is given in each "
        "panel header. Log2 axis.\n"
        "Stars: Welch t-test on field values vs the pooled controls of the same batch, "
        "Benjamini-Hochberg across all 35 condition-blocks. *** q<0.001  ** q<0.01  * q<0.05  "
        "ns not significant.\n"
        f"{caveat}\n{stamp}"
    )
    if clipped:
        footer += f"\nRed open triangles at the axis floor are {clipped} field(s) measuring exactly zero; their true fold-change is 0 and cannot be drawn on a log axis."
    fig.text(0.5, 0.008, footer, ha="center", va="bottom", fontsize=8.6, linespacing=1.5)
    fig.tight_layout(rect=(0.02, 0.085, 1, 0.975))

    png, pdf = out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf), "y_min": ymin, "y_max": ymax,
            "fields_at_zero": clipped}


# --------------------------------------------------------------------------- #
def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help="directory holding TSIII/ TSIV/ TSV/ measurement CSVs")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    metrics = ["mean_53BP1_foci_per_nucleus", "percent_nuclei_gt5_53BP1"]
    nuclei = load(args.data, "nucleus_measurements.csv")
    images = load(args.data, "image_summary.csv")
    fields, excluded = field_table(nuclei, images)
    fields = add_fold_changes(fields, metrics)

    mad = json.loads((args.data / "TSIII" / "resolved_config.json").read_text())
    mad = mad["focus_a"]["threshold_mad_multiplier"]
    stamp = (f"make_publication_53bp1_foldchange.py @ {git_commit()} | "
             f"threshold_mad_multiplier = {mad} | source: {args.data.parent.name}/{args.data.name} | "
             f"generated {date.today().isoformat()}")

    eff_mean = effects(fields, "mean_53BP1_foci_per_nucleus")
    eff_pct = effects(fields, "percent_nuclei_gt5_53BP1")

    manifest = {"stamp": stamp, "n_fields": int(len(fields)),
                "n_nuclei": int(len(nuclei)), "n_excluded_fields": int(len(excluded)),
                "batch_columns": ["experiment", "timepoint_hr"], "figures": {}}

    manifest["figures"]["fold_change_mean_53BP1"] = plot_fold_change(
        fields, eff_mean, "mean_53BP1_foci_per_nucleus",
        title="53BP1 foci: fold-change vs within-batch control",
        ylabel="Field mean 53BP1 foci per nucleus\n(fold-change vs batch control)",
        baseline_fmt="control baseline = {baseline:.2f} foci/nucleus",
        caveat=("Endpoint: field mean, the continuous measure. Robust to the detection "
                "threshold - fold-changes move <= 1.23x (median) across MAD multiplier 3-5, "
                "while absolute counts move 1.32x-0.75x. Between-drug ranking is NOT robust; "
                "read each drug against its own control, not against another drug."),
        stamp=stamp, out_dir=figures, stem="fig1_foldchange_mean_53BP1_by_batch")

    manifest["figures"]["fold_change_percent_gt5"] = plot_fold_change(
        fields, eff_pct, "percent_nuclei_gt5_53BP1",
        title="53BP1-positive nuclei (>5 foci): fold-change vs within-batch control",
        ylabel="Nuclei with >5 53BP1 foci\n(fold-change vs batch control)",
        baseline_fmt="control baseline = {baseline:.1f}% positive",
        caveat=("Endpoint: fraction of nuclei above a fixed cut-off of 5 foci. SECONDARY - a "
                "fixed cut-off is itself threshold-dependent, and these ratios spread ~1.4-1.6x "
                "across MAD multiplier 3-5 against ~1.08-1.18x for the mean. TS V control "
                "baselines are low (2.8% and 4.5%), so its fold-changes rest on a small "
                "denominator and are correspondingly unstable."),
        stamp=stamp, out_dir=figures, stem="fig2_foldchange_percent_gt5_53BP1_by_batch")

    fields.to_csv(tables / "field_level_53BP1_foldchange.csv", index=False)
    eff_mean.assign(metric="mean_53BP1_foci_per_nucleus").to_csv(
        tables / "effects_mean_53BP1.csv", index=False)
    eff_pct.assign(metric="percent_nuclei_gt5_53BP1").to_csv(
        tables / "effects_percent_gt5_53BP1.csv", index=False)
    excluded.to_csv(tables / "excluded_fields_zero_approved_nuclei.csv", index=False)
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(stamp)
    print(f"fields={len(fields)} nuclei={len(nuclei)} excluded_fields={len(excluded)}")
    for label, eff in [("mean foci", eff_mean), ("percent >5", eff_pct)]:
        sig = int((eff["q_bh"] < 0.05).sum())
        print(f"  {label:<11} {sig}/{len(eff)} condition-blocks significant at q<0.05; "
              f"fold-change {eff.fold_change.min():.2f}-{eff.fold_change.max():.2f}")
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
