"""One-command pipeline: segment every experiment, combine, and report.

The combine step is the reason this module exists. The published analysis
depended on a combined cross-experiment table that no script in the repository
produced, so the mapping from plate codes to treatment names and the
control/treated assignment were not reproducible. Here they are derived from the
config and recorded in a run manifest.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .analysis import analyse_files
from .batch import batch_aware_summary
from .config import FilenameConfig, config_to_dict
from .run_config import ExperimentSpec, RunSpec


def resolve_pattern(root: Path, patterns: list[str]) -> tuple[str, list[Path]]:
    """Pick the filename pattern that matches most files under ``root``.

    Experiments each use one naming convention, but conventions differ between
    experiments, so the config carries a list and the right one is chosen here.
    This keeps parse_sample_metadata's single-regex contract untouched.
    """
    czis = sorted(root.rglob("*.czi"))
    if not czis:
        raise SystemExit(f"No CZI files found under {root}")
    best, best_files = None, []
    for pattern in patterns:
        rx = re.compile(pattern, re.IGNORECASE)
        matched = [p for p in czis if rx.match(p.name)]
        if len(matched) > len(best_files):
            best, best_files = pattern, matched
    if not best_files:
        raise SystemExit(
            f"No file under {root} matched any configured pattern.\n"
            f"  patterns: {patterns}\n  example filename: {czis[0].name}"
        )
    if len(best_files) < len(czis):
        print(f"  note: {len(czis) - len(best_files)} of {len(czis)} CZI files did not match "
              f"the chosen pattern and are excluded")
    return best, best_files


def metric_names(spec: ExperimentSpec) -> list[str]:
    """Derived from the focus names so config cannot disagree with the columns."""
    a, b = spec.analysis.focus_a.name, spec.analysis.focus_b.name
    return [f"{a}_count", f"{b}_count", f"colocalized_{b}_count"]


def run_experiment(spec: ExperimentSpec, out_dir: Path) -> Path:
    pattern, files = resolve_pattern(spec.root, spec.patterns)
    print(f"\n=== {spec.name}: {len(files)} files ===")
    print(f"  root    : {spec.root}")
    print(f"  pattern : {pattern}")
    config = replace(spec.analysis, filename=FilenameConfig(regex=pattern))
    config_path = out_dir / "resolved_experiment_config.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config_to_dict(config), indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    analyse_files(files, config, config_path, out_dir, f"{spec.name} nuclear foci QC")
    return out_dir


def combine(run: RunSpec, per_experiment: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate experiments, mapping condition codes to names and control flags."""
    nuc_frames, img_frames = [], []
    for spec in run.experiments:
        d = per_experiment[spec.name]
        nuc = pd.read_csv(d / "nucleus_measurements.csv")
        img = pd.read_csv(d / "image_summary.csv")
        for frame in (nuc, img):
            frame["experiment"] = spec.name
            frame["condition_code"] = frame["condition"].astype(str)
            resolved = frame["condition_code"].map(lambda c: spec.resolve_condition(c))
            frame["condition"] = [r[0] for r in resolved]
            frame["known_control"] = [r[1] for r in resolved]
        nuc_frames.append(nuc)
        img_frames.append(img)

    nuclei = pd.concat(nuc_frames, ignore_index=True)
    images = pd.concat(img_frames, ignore_index=True)

    low = images[images["nuclei_count"] < run.min_nuclei_per_field]
    if len(low):
        print(f"\n  excluding {len(low)} field(s) with < {run.min_nuclei_per_field} nuclei: "
              f"{sorted(low['sample_id'])}")
        keep = set(images.loc[images["nuclei_count"] >= run.min_nuclei_per_field, "sample_id"])
        images = images[images["sample_id"].isin(keep)]
        nuclei = nuclei[nuclei["sample_id"].isin(keep)]
    return nuclei, images


def write_manifest(run: RunSpec, per_experiment: dict[str, Path], nuclei: pd.DataFrame, path: Path) -> None:
    import numpy, pandas, scipy, skimage
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parent).stdout.strip() or None
    except Exception:
        commit = None
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(run.source_path),
        "config_contents": run.raw,
        "git_commit": commit,
        "python": platform.python_version(),
        "packages": {"numpy": numpy.__version__, "pandas": pandas.__version__,
                     "scipy": scipy.__version__, "scikit-image": skimage.__version__},
        "experiments": {name: str(d) for name, d in per_experiment.items()},
        "n_nuclei": int(len(nuclei)),
        "n_fields": int(nuclei["sample_id"].nunique()),
        "batch_columns": run.batch_columns,
        "zero_inflated_metrics": run.zero_inflated_metrics,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def proportion_summary(nuclei: pd.DataFrame, metrics: list[str], batch_cols: list[str]) -> pd.DataFrame:
    """Fraction of nuclei with >=1 focus, per batch and condition.

    Preferred over control-z for zero-inflated metrics: when most control nuclei
    have zero foci the MAD is 0 and robust_control_stats silently falls back to a
    non-robust standard deviation.
    """
    rows = []
    for key, g in nuclei.groupby(batch_cols + ["condition", "known_control"], dropna=False, sort=False):
        base = dict(zip(batch_cols + ["condition", "known_control"], key if isinstance(key, tuple) else (key,)))
        row = {**base, "n_nuclei": int(len(g))}
        for m in metrics:
            row[f"fraction_ge1_{m}"] = float((g[m] >= 1).mean())
            row[f"mean_{m}"] = float(g[m].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_pipeline(config_path: Path, overwrite: bool = False) -> Path:
    run = load_and_check(config_path, overwrite)
    out = run.output_root
    out.mkdir(parents=True, exist_ok=True)

    per_experiment: dict[str, Path] = {}
    for spec in run.experiments:
        per_experiment[spec.name] = run_experiment(spec, out / "experiments" / spec.name.replace(" ", "_"))
    if not run.combine:
        print(f"\nWrote per-experiment outputs to {out}")
        return out

    nuclei, images = combine(run, per_experiment)
    combined = out / "combined"
    combined.mkdir(parents=True, exist_ok=True)
    nuclei.to_csv(combined / "combined_nucleus_measurements.csv", index=False)
    images.to_csv(combined / "combined_image_summary.csv", index=False)
    print(f"\n  combined: {len(nuclei)} nuclei across {nuclei['sample_id'].nunique()} fields")

    metrics = metric_names(run.experiments[0])
    controls = set(nuclei.loc[nuclei["known_control"], "condition"])
    print(f"  controls: {sorted(controls)}")

    if run.batch_aware_report:
        report = out / "batch_aware"
        report.mkdir(parents=True, exist_ok=True)
        for name, table in batch_aware_summary(nuclei, metrics, controls, run.batch_columns).items():
            table.to_csv(report / f"{name}.csv", index=False)
        proportion_summary(nuclei, metrics, run.batch_columns).to_csv(
            report / "proportion_positive_by_condition.csv", index=False)
        print(f"  batch-aware tables -> {report}")

    write_manifest(run, per_experiment, nuclei, out / "run_manifest.json")
    print(f"\nWrote pipeline outputs to {out}")
    return out


def load_and_check(config_path: Path, overwrite: bool) -> RunSpec:
    from .run_config import load_run_config
    run = load_run_config(config_path)
    if run.output_root.exists() and any(run.output_root.iterdir()) and not overwrite:
        raise SystemExit(
            f"Refusing to write into a non-empty output directory: {run.output_root}\n"
            f"Choose a new run.output_root, or pass --overwrite to replace it."
        )
    if overwrite and run.output_root.exists():
        import shutil
        shutil.rmtree(run.output_root)
    return run
