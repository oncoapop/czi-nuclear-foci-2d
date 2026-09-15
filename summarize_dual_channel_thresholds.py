#!/usr/bin/env python3
"""Summarise per-nucleus dual-channel thresholds by timepoint and condition."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CONTROL_CONDITIONS = {"Aqueous", "DMSO", "NAH2PO4"}
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


def threshold_performance(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    treated = ~df["condition"].isin(CONTROL_CONDITIONS)
    controls = df["condition"].isin(CONTROL_CONDITIONS)
    rows = []
    for threshold in range(0, int(df[metric].max()) + 2):
        positive = df[metric] >= threshold
        tp = int((positive & treated).sum())
        fp = int((positive & controls).sum())
        tn = int((~positive & controls).sum())
        fn = int((~positive & treated).sum())
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append(
            {
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


def summary_rows(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_cols, observed=True, sort=True)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_cols, group_key))
        n = len(group)
        row.update(
            {
                "n_nuclei": n,
                "mean_af488_spots_per_nucleus": group["af488_spots_count"].mean(),
                "median_af488_spots_per_nucleus": group["af488_spots_count"].median(),
                "nuclei_af488_ge5": int((group["af488_spots_count"] >= 5).sum()),
                "fraction_nuclei_af488_ge5": float((group["af488_spots_count"] >= 5).mean()),
                "mean_rhrex_spots_per_nucleus": group["rhrex_spots_count"].mean(),
                "median_rhrex_spots_per_nucleus": group["rhrex_spots_count"].median(),
                "nuclei_rhrex_ge1": int((group["rhrex_spots_count"] >= 1).sum()),
                "fraction_nuclei_rhrex_ge1": float((group["rhrex_spots_count"] >= 1).mean()),
                "mean_colocalized_rhrex_spots_per_nucleus": group["colocalized_rhrex_spots_count"].mean(),
                "median_colocalized_rhrex_spots_per_nucleus": group["colocalized_rhrex_spots_count"].median(),
                "nuclei_colocalized_ge1": int((group["colocalized_rhrex_spots_count"] >= 1).sum()),
                "fraction_nuclei_colocalized_ge1": float((group["colocalized_rhrex_spots_count"] >= 1).mean()),
                "nuclei_colocalized_gt5": int((group["colocalized_rhrex_spots_count"] > 5).sum()),
                "fraction_nuclei_colocalized_gt5": float((group["colocalized_rhrex_spots_count"] > 5).mean()),
                "max_colocalized_rhrex_spots_per_nucleus": int(group["colocalized_rhrex_spots_count"].max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    df["timepoint_hr"] = df["timepoint_hr"].astype(int)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)

    by_time_condition = summary_rows(df, ["timepoint_hr", "condition"])
    by_time = summary_rows(df, ["timepoint_hr"])
    pooled = summary_rows(df, ["condition"])

    by_time_condition.to_csv(args.output_dir / "summary_by_time_condition.csv", index=False)
    by_time.to_csv(args.output_dir / "summary_by_time.csv", index=False)
    pooled.to_csv(args.output_dir / "summary_by_condition_pooled.csv", index=False)

    evidence_frames = []
    best_rows = []
    for metric in ["af488_spots_count", "rhrex_spots_count", "colocalized_rhrex_spots_count"]:
        perf = threshold_performance(df, metric)
        perf.to_csv(args.output_dir / f"threshold_evidence_{metric}.csv", index=False)
        evidence_frames.append(perf)
        best = perf.sort_values(["youden_j", "specificity", "sensitivity"], ascending=False).iloc[0].to_dict()
        best_rows.append(best)
    pd.DataFrame(best_rows).to_csv(args.output_dir / "best_thresholds_by_metric.csv", index=False)

    # Explicit user-requested strict >5 co-localised foci evidence.
    coloc_gt5 = df.assign(colocalized_gt5=df["colocalized_rhrex_spots_count"] > 5)
    gt5 = (
        coloc_gt5.groupby(["timepoint_hr", "condition"], observed=True)
        .agg(
            n_nuclei=("colocalized_gt5", "size"),
            nuclei_colocalized_gt5=("colocalized_gt5", "sum"),
            fraction_nuclei_colocalized_gt5=("colocalized_gt5", "mean"),
        )
        .reset_index()
    )
    gt5.to_csv(args.output_dir / "explicit_colocalized_gt5_by_time_condition.csv", index=False)

    print(f"Wrote summaries to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
