#!/usr/bin/env python3
"""53BP1 effect sizes per condition, with the MAD multiplier as a robustness axis.

Unit of analysis is the imaging field, not the nucleus: nuclei within a field are
not independent, and treating them as replicates inflates significance. Each
treatment is compared with the pooled controls of its own experiment x timepoint
block, so batch and timepoint baseline differences are never crossed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AUDIT = Path("/Volumes/Backup/czi-audit-2026-09-15")
SWEEP = AUDIT / "sweep53"
EXPERIMENTS = {"tsIII": "TS III", "tsIV": "TS IV", "tsV": "TS V"}
MADS = ["3.0", "4.0", "5.0"]
CONTROLS = {"Aqueous", "DMSO", "NAH2PO4", "Media"}
TSV_MAP = {"A": "CX-5461", "B": "Cisplatin", "C": "Etoposide", "D": "Palbociclib",
           "E": "NAH2PO4", "F": "Media", "G": "DMSO", "H": "PDS"}
NON_GENOTOXIC = "Palbociclib"
POSITIVE_ANCHOR = "Etoposide"


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return out


def load(mad: str) -> pd.DataFrame:
    frames = []
    for key, label in EXPERIMENTS.items():
        d = SWEEP / f"{key}_mad{mad}"
        n = pd.read_csv(d / "nucleus_measurements.csv")
        n["experiment"] = label
        n["condition"] = (n["condition"].map(TSV_MAP).fillna(n["condition"])
                          if label == "TS V" else n["condition"])
        frames.append(n)
    n = pd.concat(frames, ignore_index=True)
    n["timepoint_hr"] = n["timepoint_hr"].astype(int)
    n["is_control"] = n["condition"].isin(CONTROLS)
    return n


def field_effects(nuclei: pd.DataFrame) -> pd.DataFrame:
    fields = (nuclei.groupby(["experiment", "timepoint_hr", "condition", "is_control", "sample_id"])
                    ["53BP1_count"].mean().reset_index(name="field_mean"))
    rows = []
    for (exp, tp), block in fields.groupby(["experiment", "timepoint_hr"]):
        ctl = block.loc[block.is_control, "field_mean"]
        if len(ctl) < 2:
            continue
        for cond, g in block[~block.is_control].groupby("condition"):
            if len(g) < 2:
                continue
            t = stats.ttest_ind(g.field_mean, ctl, equal_var=False)
            pooled = np.sqrt((g.field_mean.var(ddof=1) + ctl.var(ddof=1)) / 2)
            rows.append({
                "experiment": exp, "timepoint_hr": tp, "condition": cond,
                "n_treated_fields": len(g), "n_control_fields": len(ctl),
                "control_mean": ctl.mean(), "treated_mean": g.field_mean.mean(),
                "fold_change": g.field_mean.mean() / ctl.mean() if ctl.mean() else np.nan,
                "cohens_d": (g.field_mean.mean() - ctl.mean()) / pooled if pooled else np.nan,
                "p_welch": t.pvalue,
            })
    out = pd.DataFrame(rows)
    out["q_bh"] = benjamini_hochberg(out["p_welch"].to_numpy())
    return out.sort_values(["experiment", "timepoint_hr", "fold_change"], ascending=[True, True, False])


def benchmark(eff: pd.DataFrame) -> dict:
    blocks = eff.groupby(["experiment", "timepoint_hr"])
    anchor_top = sum(g.sort_values("fold_change", ascending=False).iloc[0].condition == POSITIVE_ANCHOR
                     for _, g in blocks)
    anchor_sig = sum(((g.condition == POSITIVE_ANCHOR) & (g.q_bh < 0.05)).any() for _, g in blocks)
    neg = eff[eff.condition == NON_GENOTOXIC]
    return {"blocks": blocks.ngroups,
            "anchor_ranked_first": anchor_top,
            "anchor_significant": anchor_sig,
            "negative_control_null": int((neg.q_bh >= 0.05).sum()),
            "negative_control_blocks": len(neg)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=AUDIT / "effects53")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_effects = []
    for mad in MADS:
        eff = field_effects(load(mad)).assign(mad_multiplier=float(mad))
        eff.to_csv(args.output_dir / f"effects_mad{mad}.csv", index=False)
        b = benchmark(eff)
        print(f"\nMAD {mad}:  {POSITIVE_ANCHOR} ranked #1 in {b['anchor_ranked_first']}/{b['blocks']} blocks, "
              f"significant (q<0.05) in {b['anchor_significant']}/{b['blocks']}  |  "
              f"{NON_GENOTOXIC} indistinguishable from vehicle in "
              f"{b['negative_control_null']}/{b['negative_control_blocks']}")
        all_effects.append(eff)

    combined = pd.concat(all_effects, ignore_index=True)
    combined.to_csv(args.output_dir / "effects_all_mad.csv", index=False)

    wide = combined.pivot_table(index=["experiment", "timepoint_hr", "condition"],
                                columns="mad_multiplier", values="fold_change")
    wide.columns = [f"fold_MAD{c}" for c in wide.columns]
    wide["min"] = wide.min(axis=1); wide["max"] = wide.max(axis=1)
    wide["spread"] = wide["max"] / wide["min"]
    wide["direction_stable"] = (wide[[c for c in wide.columns if c.startswith("fold_")]] > 1).all(axis=1) | \
                               (wide[[c for c in wide.columns if c.startswith("fold_")]] < 1).all(axis=1)
    wide.round(3).to_csv(args.output_dir / "fold_change_robustness.csv")

    print("\n" + "=" * 92)
    print("53BP1 fold-change vs within-block control, across the parameter range")
    print("=" * 92)
    print(wide.round(2).to_string())
    print(f"\n  direction stable in {int(wide.direction_stable.sum())}/{len(wide)} condition-blocks")
    print(f"  median spread across MAD 3-5: {wide.spread.median():.2f}x   max: {wide.spread.max():.2f}x")
    print(f"\nwrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
