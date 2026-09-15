#!/usr/bin/env python3
"""Compare a re-run dual-channel segmentation output against the published one.

Per-image thresholds are computed independently for each field, so re-running any
subset of files must reproduce the published per-field numbers exactly. Any
difference is a real finding, not noise.

Usage:
  audit_diff_repro.py --rerun DIR --published DIR [--published DIR ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


IMAGE_CSV = "image_summary_dual_channel.csv"
NUCLEUS_CSV = "nucleus_dual_channel_measurements.csv"

IMAGE_KEY = ["sample_id"]
NUCLEUS_KEY = ["sample_id", "nucleus_label"]

# Columns that legitimately differ between runs and are not evidence of drift.
IGNORE = {"source_path"}


def load(dirs: list[Path], name: str) -> pd.DataFrame:
    frames = []
    for d in dirs:
        path = d / name
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def compare(rerun: pd.DataFrame, published: pd.DataFrame, key: list[str], label: str) -> int:
    rerun = rerun.sort_values(key).reset_index(drop=True)
    published = published[published["sample_id"].isin(set(rerun["sample_id"]))]
    published = published.sort_values(key).reset_index(drop=True)

    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(f"  re-run rows    : {len(rerun)}")
    print(f"  published rows : {len(published)}  (restricted to re-run sample_ids)")

    r_keys = set(map(tuple, rerun[key].values))
    p_keys = set(map(tuple, published[key].values))
    only_rerun, only_pub = r_keys - p_keys, p_keys - r_keys
    if only_rerun or only_pub:
        print(f"  !! key mismatch: {len(only_rerun)} only in re-run, {len(only_pub)} only in published")
        for k in list(only_rerun)[:5]:
            print(f"       only re-run   : {k}")
        for k in list(only_pub)[:5]:
            print(f"       only published: {k}")
        return 1
    print(f"  keys           : identical ({len(r_keys)})")

    shared = [c for c in rerun.columns if c in published.columns and c not in IGNORE]
    missing_cols = (set(rerun.columns) ^ set(published.columns)) - IGNORE
    if missing_cols:
        print(f"  note: columns present in only one side (skipped): {sorted(missing_cols)}")

    failures = 0
    for col in shared:
        a, b = rerun[col], published[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            both_nan = np.isnan(av) & np.isnan(bv)
            equal = (av == bv) | both_nan
            if equal.all():
                continue
            n_bad = int((~equal).sum())
            delta = np.nanmax(np.abs(av[~equal] - bv[~equal]))
            print(f"  MISMATCH {col}: {n_bad}/{len(av)} rows differ, max |delta| = {delta:.6g}")
            for idx in np.flatnonzero(~equal)[:5]:
                keyval = tuple(rerun.loc[idx, k] for k in key)
                print(f"       {keyval}: re-run={av[idx]!r} published={bv[idx]!r}")
            failures += 1
        else:
            equal = (a.astype(str) == b.astype(str))
            if equal.all():
                continue
            n_bad = int((~equal).sum())
            print(f"  MISMATCH {col}: {n_bad}/{len(a)} rows differ (text)")
            for idx in np.flatnonzero(~equal.to_numpy())[:3]:
                print(f"       {tuple(rerun.loc[idx, k] for k in key)}: "
                      f"re-run={a.iloc[idx]!r} published={b.iloc[idx]!r}")
            failures += 1

    if failures == 0:
        print(f"  RESULT         : IDENTICAL across all {len(shared)} compared columns")
    else:
        print(f"  RESULT         : {failures} column(s) differ")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rerun", type=Path, required=True)
    ap.add_argument("--published", type=Path, action="append", required=True)
    args = ap.parse_args()

    total = 0
    total += compare(load([args.rerun], IMAGE_CSV), load(args.published, IMAGE_CSV),
                     IMAGE_KEY, "PER-IMAGE SUMMARY")
    total += compare(load([args.rerun], NUCLEUS_CSV), load(args.published, NUCLEUS_CSV),
                     NUCLEUS_KEY, "PER-NUCLEUS MEASUREMENTS")

    print(f"\n{'=' * 78}")
    print("VERDICT: reproduction is BIT-EXACT" if total == 0
          else f"VERDICT: {total} differing column group(s) - NOT reproduced")
    print(f"{'=' * 78}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
