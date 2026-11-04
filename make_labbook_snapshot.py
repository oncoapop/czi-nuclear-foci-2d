#!/usr/bin/env python3
"""Assemble a self-contained lab-book snapshot of the 2-D 53BP1 analysis.

Gathers, into one directory outside any git checkout:

  - the fold-change figures and the absolute-value figures they supersede;
  - every analysis table, per field and per condition-block;
  - the underlying per-nucleus and per-focus CSVs for all three experiments;
  - the segmentation masks (nuclei labels and 53BP1 focus labels) and the
    per-field QC overlays, so a count can be traced back to the pixels;
  - the scripts and configs that produced all of it.

Nuclei mask integrity is verified against the sha256 recorded in each
experiment's `approved_nuclei_mask_manifest.csv` at copy time. Any mismatch is
reported and fails the run - a snapshot that silently carries a changed mask is
worse than no snapshot.

Everything is additive: the script creates a new directory and never modifies
the sources it reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

AUDIT = Path(__file__).resolve().parent
LEGACY = Path("/Users/dyap/Documents/Tasks workflow/czi-analysis")
DATA = LEGACY / "output" / "publication_53BP1_v1" / "data"
FOLDCHANGE = AUDIT / "output" / "publication_53BP1_foldchange_v1"
SUPERSEDED = LEGACY / "output" / "publication_53BP1_gt5_v1"
SEGMENTATION = {
    "TSIII": LEGACY / "output/segmentation/time_series_III_dual_channel_coloc/masks",
    "TSIV": LEGACY / "output/segmentation/time_series_IV_dual_channel_coloc/masks",
    "TSV": LEGACY / "output/segmentation/time_series_V_dual_channel_coloc/masks",
}
EXPERIMENTS = ["TSIII", "TSIV", "TSV"]
PER_EXPERIMENT_FILES = [
    "nucleus_measurements.csv",
    "image_summary.csv",
    "53BP1_focus_measurements.csv",
    "approved_nuclei_mask_manifest.csv",
    "resolved_config.json",
    "STATUS.txt",
    "53BP1_qc_contact_sheet.pdf",
]
CODE_FILES = [
    AUDIT / "make_publication_53bp1_foldchange.py",
    AUDIT / "make_labbook_snapshot.py",
    LEGACY / "make_publication_53bp1.py",
    AUDIT / "segment_dual_channel_coloc.py",
    AUDIT / "segment_nuclei_spots.py",
    AUDIT / "configs" / "time_series.yaml",
    AUDIT / "docs" / "PROVENANCE.md",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: Path, dst: Path) -> int:
    if not src.is_dir():
        return 0
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return sum(1 for p in dst.rglob("*") if p.is_file())


def copy_nuclei_masks(experiment: str, manifest_csv: Path, dst: Path) -> dict:
    """Copy the approved nuclei label TIFFs, checking each against its sha256."""
    dst.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_csv)
    source_dir = SEGMENTATION[experiment]
    copied, mismatched, missing = 0, [], []
    for _, row in manifest.iterrows():
        src = source_dir / Path(row["approved_nuclei_mask_path"]).name
        if not src.is_file():
            missing.append(src.name)
            continue
        digest = sha256(src)
        if digest != row["approved_nuclei_mask_sha256"]:
            mismatched.append({"file": src.name,
                               "expected": row["approved_nuclei_mask_sha256"],
                               "found": digest})
            continue
        shutil.copy2(src, dst / src.name)
        copied += 1
    return {"copied": copied, "verified": copied,
            "mismatched": mismatched, "missing": missing,
            "expected": int(len(manifest))}


def git_commit(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path,
                    default=Path("/Volumes/Backup/czi-analysis-runs")
                    / f"{date.today().isoformat()}_53BP1_2D_labbook_snapshot")
    ap.add_argument("--skip-masks", action="store_true",
                    help="tables, figures and QC images only (~250 MB instead of ~1.2 GB)")
    args = ap.parse_args()

    out = args.output
    if out.exists():
        raise SystemExit(f"refusing to overwrite an existing snapshot: {out}")
    out.mkdir(parents=True)
    print(f"snapshot -> {out}\n")

    manifest: dict = {
        "snapshot_date": date.today().isoformat(),
        "audit_repo_commit": git_commit(AUDIT),
        "legacy_repo_commit": git_commit(LEGACY),
        "masks_included": not args.skip_masks,
        "sources": {"measurements": str(DATA),
                    "fold_change_output": str(FOLDCHANGE),
                    "superseded_output": str(SUPERSEDED)},
        "experiments": {},
    }

    # figures -------------------------------------------------------------
    figures = out / "figures"
    (figures / "fold_change").mkdir(parents=True)
    (figures / "absolute_SUPERSEDED").mkdir(parents=True)
    for item in (FOLDCHANGE / "figures").iterdir():
        shutil.copy2(item, figures / "fold_change" / item.name)
    for item in (SUPERSEDED / "figures").iterdir():
        shutil.copy2(item, figures / "absolute_SUPERSEDED" / item.name)
    print(f"  figures            fold_change={len(list((figures / 'fold_change').iterdir()))} "
          f"superseded={len(list((figures / 'absolute_SUPERSEDED').iterdir()))}")

    # tables --------------------------------------------------------------
    tables = out / "tables"
    tables.mkdir()
    for item in (FOLDCHANGE / "tables").iterdir():
        shutil.copy2(item, tables / item.name)
    shutil.copy2(FOLDCHANGE / "run_manifest.json", tables / "foldchange_run_manifest.json")
    shutil.copy2(FOLDCHANGE / "README.md", out / "ANALYSIS_NOTES.md")
    for item in (SUPERSEDED / "tables").iterdir():
        shutil.copy2(item, tables / f"SUPERSEDED_{item.name}")
    print(f"  tables             {len(list(tables.iterdir()))} files")

    # per-experiment data -------------------------------------------------
    for experiment in EXPERIMENTS:
        src, dst = DATA / experiment, out / "data" / experiment
        dst.mkdir(parents=True)
        for name in PER_EXPERIMENT_FILES:
            if (src / name).is_file():
                shutil.copy2(src / name, dst / name)
        entry: dict = {"qc_images": copy_tree(src / "qc_images", dst / "qc_images")}
        if not args.skip_masks:
            entry["focus_masks_53BP1"] = copy_tree(src / "focus_masks", dst / "focus_masks_53BP1")
            entry["nuclei_masks"] = copy_nuclei_masks(
                experiment, src / "approved_nuclei_mask_manifest.csv", dst / "nuclei_masks")
        manifest["experiments"][experiment] = entry

        nuclei = entry.get("nuclei_masks")
        line = (f"  {experiment:<6} qc={entry['qc_images']:>3}"
                f"  focus_masks={entry.get('focus_masks_53BP1', 0):>3}")
        if nuclei:
            line += (f"  nuclei_masks={nuclei['copied']:>3}"
                     f"  (sha256 verified {nuclei['verified']}/{nuclei['expected']})")
        print(line)

    # code ----------------------------------------------------------------
    code = out / "code"
    code.mkdir()
    for path in CODE_FILES:
        if path.is_file():
            shutil.copy2(path, code / path.name)
    print(f"  code               {len(list(code.iterdir()))} files")

    # integrity -----------------------------------------------------------
    problems = []
    for experiment, entry in manifest["experiments"].items():
        nuc = entry.get("nuclei_masks")
        if not nuc:
            continue
        if nuc["mismatched"]:
            problems.append(f"{experiment}: {len(nuc['mismatched'])} nuclei mask(s) "
                            f"failed sha256")
        if nuc["missing"]:
            problems.append(f"{experiment}: {len(nuc['missing'])} nuclei mask(s) not found")

    digests = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.suffix.lower() not in {".tif", ".tiff"}:
            digests[str(path.relative_to(out))] = sha256(path)
    (out / "CHECKSUMS.sha256").write_text(
        "".join(f"{d}  {name}\n" for name, d in sorted(digests.items())))

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    manifest["total_bytes"] = total
    manifest["total_files"] = sum(1 for p in out.rglob("*") if p.is_file())
    manifest["integrity_problems"] = problems
    (out / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n  {manifest['total_files']} files, {total / 1e9:.2f} GB")
    if problems:
        print("\nINTEGRITY PROBLEMS:")
        for line in problems:
            print(f"  - {line}")
        return 1
    print("  all nuclei masks matched their recorded sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
