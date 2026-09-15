"""Command line interface for generic CZI nuclear foci analysis."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

from .analysis import analyse_files
from .batch import batch_aware_summary
from .config import load_config
from .current_dataset import run_current_dataset_report
from .focus_qc import analyse_focus_qc_with_masks, analyse_single_focus_with_masks
from .io import selected_files_from_manifest, selected_files_from_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="czi-foci", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyse = subparsers.add_parser("analyse", help="Segment nuclei, two focus channels, and co-localised foci")
    source = analyse.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path, help="Root folder containing CZI files")
    source.add_argument("--manifest", type=Path, help="CSV manifest with a source_path column")
    analyse.add_argument("--config", type=Path, required=True, help="JSON analysis configuration")
    analyse.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    analyse.add_argument("--qc-title", default="CZI nuclear foci QC")
    analyse.add_argument("--limit", type=int, help="Process only the first N selected files")
    analyse.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")

    focus_qc = subparsers.add_parser("focus-qc", help="Detect two focus channels using fixed approved nuclei masks; no co-localisation")
    focus_source = focus_qc.add_mutually_exclusive_group(required=True)
    focus_source.add_argument("--root", type=Path, help="Root folder containing CZI files")
    focus_source.add_argument("--manifest", type=Path, help="CSV manifest with a source_path column")
    focus_qc.add_argument("--config", type=Path, required=True, help="JSON analysis configuration")
    focus_qc.add_argument("--nuclei-mask-dir", type=Path, required=True, help="Directory containing approved <sample_id>_nuclei_labels.tif files")
    focus_qc.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    focus_qc.add_argument("--qc-title", default="Focus detection QC using approved nuclei masks")
    focus_qc.add_argument("--limit", type=int, help="Process only the first N selected files")
    focus_qc.add_argument("--replicate", help="Process only this filename replicate/field, for example 01")

    single = subparsers.add_parser("single-focus", help="Analyse focus A only using fixed approved nuclei masks")
    single_source = single.add_mutually_exclusive_group(required=True)
    single_source.add_argument("--root", type=Path, help="Root folder containing CZI files")
    single_source.add_argument("--manifest", type=Path, help="CSV manifest with a source_path column")
    single.add_argument("--config", type=Path, required=True, help="JSON analysis configuration; focus_a is analysed")
    single.add_argument("--nuclei-mask-dir", type=Path, required=True)
    single.add_argument("--output-dir", type=Path, required=True)
    single.add_argument("--qc-title", default="Single-focus QC")
    single.add_argument("--limit", type=int)

    batch = subparsers.add_parser("batch-summary", help="Compute batch-aware normalisation and thresholds")
    batch.add_argument("--nucleus-csv", type=Path, action="append", required=True, help="Input nucleus_measurements.csv; repeat for multiple experiments")
    batch.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    batch.add_argument("--metric", action="append", required=True, help="Per-nucleus metric column; repeat for multiple metrics")
    batch.add_argument("--control-condition", action="append", required=True, help="Control condition name; repeat for multiple controls")
    batch.add_argument("--batch-column", action="append", default=[], help="Batch column; defaults to experiment and timepoint_hr when present")
    batch.add_argument("--experiment-label", action="append", default=[], help="Optional experiment labels matching --nucleus-csv order")
    batch.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")

    current = subparsers.add_parser("current-dataset-report", help="Report batch-aware thresholds for the existing TS III/IV/V dataset")
    current.add_argument(
        "--nucleus-csv",
        type=Path,
        default=Path("output/segmentation/all_experiments_dual_channel_summary/combined_nucleus_dual_channel_measurements.csv"),
        help="Combined mapped nucleus table",
    )
    current.add_argument(
        "--image-csv",
        type=Path,
        default=Path("output/segmentation/all_experiments_dual_channel_summary/combined_image_summary_dual_channel.csv"),
        help="Combined mapped image summary table",
    )
    current.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/segmentation/batch_aware_current_dataset"),
        help="Output directory",
    )
    current.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    current.add_argument("--no-qc-pdf", action="store_true", help="Skip control QC PDF generation")

    plots = subparsers.add_parser("current-dataset-plots", help="Plot current dataset field summaries in the first-iteration multi-panel format")
    plots.add_argument(
        "--field-csv",
        type=Path,
        default=Path("output/segmentation/batch_aware_current_dataset/comparison_field_replicates_single_dual_coloc.csv"),
        help="Field-level comparison table",
    )
    plots.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/segmentation/batch_aware_current_dataset/plots"),
        help="Output directory for PNG/PDF plots",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyse":
        config = load_config(args.config)
        files = selected_files_from_root(args.root, config) if args.root else selected_files_from_manifest(args.manifest)
        if args.limit:
            files = files[: args.limit]
        if not files:
            parser.error("No CZI files selected")
        if args.output_dir.exists():
            if not args.overwrite:
                parser.error(f"Refusing to overwrite existing output directory: {args.output_dir}")
            shutil.rmtree(args.output_dir)
        analyse_files(files, config, args.config.resolve(), args.output_dir, args.qc_title)
        print(f"Wrote CZI foci analysis outputs to {args.output_dir}")
        return 0
    if args.command == "focus-qc":
        config = load_config(args.config)
        files = selected_files_from_root(args.root, config) if args.root else selected_files_from_manifest(args.manifest)
        if args.replicate:
            requested = args.replicate.zfill(2)
            selected = []
            for path in files:
                match = re.match(config.filename.regex, path.name, re.IGNORECASE)
                if match and match.group(config.filename.replicate_group).zfill(2) == requested:
                    selected.append(path)
            files = selected
        if args.limit:
            files = files[: args.limit]
        if not files:
            parser.error("No CZI files selected")
        if args.output_dir.exists():
            parser.error(f"Refusing to overwrite existing output directory: {args.output_dir}")
        analyse_focus_qc_with_masks(
            files, config, args.config.resolve(), args.nuclei_mask_dir, args.output_dir, args.qc_title
        )
        print(f"Wrote provisional focus-QC outputs to {args.output_dir}; no co-localisation was calculated")
        return 0
    if args.command == "single-focus":
        config = load_config(args.config)
        files = selected_files_from_root(args.root, config) if args.root else selected_files_from_manifest(args.manifest)
        if args.limit:
            files = files[: args.limit]
        if not files:
            parser.error("No CZI files selected")
        if args.output_dir.exists():
            parser.error(f"Refusing to overwrite existing output directory: {args.output_dir}")
        analyse_single_focus_with_masks(
            files, config, args.config.resolve(), args.nuclei_mask_dir, args.output_dir, args.qc_title
        )
        print(f"Wrote single-focus outputs to {args.output_dir}")
        return 0
    if args.command == "batch-summary":
        if args.output_dir.exists():
            if not args.overwrite:
                parser.error(f"Refusing to overwrite existing output directory: {args.output_dir}")
            shutil.rmtree(args.output_dir)
        args.output_dir.mkdir(parents=True)
        frames = []
        labels = args.experiment_label
        if labels and len(labels) != len(args.nucleus_csv):
            parser.error("--experiment-label must be supplied once per --nucleus-csv")
        for index, csv_path in enumerate(args.nucleus_csv):
            frame = pd.read_csv(csv_path)
            if "experiment" not in frame.columns:
                frame["experiment"] = labels[index] if labels else csv_path.parent.name
            frames.append(frame)
        nuclei = pd.concat(frames, ignore_index=True)
        batch_cols = args.batch_column or [col for col in ["experiment", "timepoint_hr"] if col in nuclei.columns]
        if not batch_cols:
            parser.error("No batch columns supplied and neither experiment nor timepoint_hr is present")
        outputs = batch_aware_summary(nuclei, args.metric, set(args.control_condition), batch_cols)
        for name, table in outputs.items():
            table.to_csv(args.output_dir / f"{name}.csv", index=False)
        print(f"Wrote batch-aware summaries to {args.output_dir}")
        return 0
    if args.command == "current-dataset-report":
        run_current_dataset_report(
            nucleus_csv=args.nucleus_csv,
            image_csv=args.image_csv,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            write_qc=not args.no_qc_pdf,
        )
        print(f"Wrote current-dataset batch-aware report to {args.output_dir}")
        return 0
    if args.command == "current-dataset-plots":
        from .current_dataset_plots import write_current_dataset_plots

        outputs = write_current_dataset_plots(args.field_csv, args.output_dir)
        print(f"Wrote current-dataset comparison plots to {args.output_dir}")
        for path in outputs:
            print(path)
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
