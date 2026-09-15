#!/usr/bin/env python3
"""Label TRF2Opt well-based analysis outputs with the experimental sample sheet."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("output/trf2opt_trf2_dapi")
DEFAULT_OUTPUT = Path("output/trf2opt_trf2_dapi_labelled")


SAMPLE_SHEET = [
    {"timepoint_hr": 2, "well": "Well1", "drug": "NaH2PO4", "trf2_dilution": "1:300"},
    {"timepoint_hr": 2, "well": "Well2", "drug": "NaH2PO4", "trf2_dilution": "1:1000"},
    {"timepoint_hr": 2, "well": "Well3", "drug": "1 uM CX5461", "trf2_dilution": "1:300"},
    {"timepoint_hr": 2, "well": "Well4", "drug": "1 uM CX5461", "trf2_dilution": "1:500"},
    {"timepoint_hr": 2, "well": "Well5", "drug": "1 uM CX5461", "trf2_dilution": "1:1000"},
    {"timepoint_hr": 2, "well": "Well6", "drug": "1 uM CX5461", "trf2_dilution": "1:2000"},
    {"timepoint_hr": 2, "well": "Well7", "drug": "NaH2PO4", "trf2_dilution": "None"},
    {"timepoint_hr": 2, "well": "Well8", "drug": "1 uM CX5461", "trf2_dilution": "None"},
    {"timepoint_hr": 24, "well": "Well1", "drug": "NaH2PO4", "trf2_dilution": "1:300"},
    {"timepoint_hr": 24, "well": "Well2", "drug": "NaH2PO4", "trf2_dilution": "1:1000"},
    {"timepoint_hr": 24, "well": "Well3", "drug": "1 uM CX5461", "trf2_dilution": "1:300"},
    {"timepoint_hr": 24, "well": "Well4", "drug": "1 uM CX5461", "trf2_dilution": "1:500"},
    {"timepoint_hr": 24, "well": "Well5", "drug": "1 uM CX5461", "trf2_dilution": "1:1000"},
    {"timepoint_hr": 24, "well": "Well6", "drug": "1 uM CX5461", "trf2_dilution": "1:2000"},
    {"timepoint_hr": 24, "well": "Well7", "drug": "1 uM CX5461", "trf2_dilution": "1:300"},
    {"timepoint_hr": 24, "well": "Well8", "drug": "1 uM CX5461", "trf2_dilution": "None"},
]


CSV_FILES = [
    "condition_summary.csv",
    "image_summary.csv",
    "nucleus_measurements.csv",
    "source_metadata.csv",
    "trf2_puncta_measurements.csv",
]


def load_map() -> pd.DataFrame:
    sample_sheet = pd.DataFrame(SAMPLE_SHEET)
    sample_sheet["condition_label"] = (
        sample_sheet["timepoint_hr"].astype(str)
        + " hr | "
        + sample_sheet["drug"]
        + " | TRF2 "
        + sample_sheet["trf2_dilution"]
    )
    sample_sheet["antibody_present"] = sample_sheet["trf2_dilution"].ne("None")
    return sample_sheet


def label_frame(frame: pd.DataFrame, sample_sheet: pd.DataFrame) -> pd.DataFrame:
    labelled = frame.copy()
    labelled["timepoint_hr"] = labelled["timepoint_hr"].astype(int)
    labelled = labelled.drop(columns=[c for c in ("drug", "trf2_dilution", "condition_label", "antibody_present") if c in labelled])
    labelled = labelled.merge(sample_sheet, on=["timepoint_hr", "well"], how="left", validate="many_to_one")
    if labelled["drug"].isna().any():
        missing = labelled.loc[labelled["drug"].isna(), ["timepoint_hr", "well"]].drop_duplicates()
        raise ValueError(f"Missing sample-sheet labels for:\n{missing.to_string(index=False)}")
    return labelled


def suitability_table(labelled_nuclei: pd.DataFrame) -> pd.DataFrame:
    valid_for_summary = labelled_nuclei["valid_for_summary"].astype(str).str.upper().eq("TRUE")
    valid = labelled_nuclei[
        valid_for_summary
        & (labelled_nuclei["antibody_present"])
    ].copy()
    grouped = valid.groupby(["timepoint_hr", "drug", "trf2_dilution", "condition_label"], sort=True)
    summary = grouped.agg(
        n_valid_nuclei=("nucleus_label", "count"),
        mean_nuclear_trf2=("nucleus_mean_trf2_intensity", "mean"),
        median_nuclear_trf2=("nucleus_mean_trf2_intensity", "median"),
        mean_background_subtracted_trf2=("nucleus_background_subtracted_mean_trf2", "mean"),
        mean_positive_area_fraction=("trf2_positive_area_fraction", "mean"),
        mean_puncta_per_nucleus=("trf2_puncta_count", "mean"),
        median_puncta_per_nucleus=("trf2_puncta_count", "median"),
    ).reset_index()

    none_controls = labelled_nuclei[
        valid_for_summary
        & (~labelled_nuclei["antibody_present"])
    ].groupby(["timepoint_hr", "drug"], sort=True).agg(
        none_control_mean_background_subtracted_trf2=("nucleus_background_subtracted_mean_trf2", "mean"),
        none_control_mean_puncta=("trf2_puncta_count", "mean"),
    ).reset_index()
    summary = summary.merge(none_controls, on=["timepoint_hr", "drug"], how="left")
    summary["fold_over_matching_none_background"] = (
        summary["mean_background_subtracted_trf2"]
        / summary["none_control_mean_background_subtracted_trf2"].replace({0: pd.NA})
    )
    summary["fold_over_matching_none_puncta"] = (
        summary["mean_puncta_per_nucleus"]
        / summary["none_control_mean_puncta"].replace({0: pd.NA})
    )

    def call(row: pd.Series) -> str:
        if row["n_valid_nuclei"] < 30:
            return "borderline - low nuclei"
        if row["mean_background_subtracted_trf2"] < 20:
            return "not suitable - weak TRF2"
        if row["mean_puncta_per_nucleus"] < 0.5:
            return "not suitable - few puncta"
        if row["mean_puncta_per_nucleus"] > 2.5:
            return "review - very punctate/possibly high background"
        if row["fold_over_matching_none_background"] < 3 if pd.notna(row["fold_over_matching_none_background"]) else False:
            return "review - weak separation from no-primary"
        return "suitable"

    summary["suitability_call"] = summary.apply(call, axis=1)
    summary["suitability_rank"] = summary["suitability_call"].map(
        {
            "suitable": 1,
            "review - very punctate/possibly high background": 2,
            "review - weak separation from no-primary": 3,
            "borderline - low nuclei": 4,
            "not suitable - few puncta": 5,
            "not suitable - weak TRF2": 6,
        }
    )
    summary = summary.sort_values(
        [
            "suitability_rank",
            "timepoint_hr",
            "drug",
            "mean_background_subtracted_trf2",
            "mean_puncta_per_nucleus",
        ],
        ascending=[True, True, True, False, False],
    ).drop(columns=["suitability_rank"])
    return summary


def write_report(path: Path, suitability: pd.DataFrame) -> None:
    calls = suitability["suitability_call"].astype(str)
    best = suitability[calls.eq("suitable")].copy()
    review = suitability[calls.str.startswith("review")].copy()
    lines = [
        "# TRF2Opt labelled suitability summary",
        "",
        "Interpretation boundary: TRF2 puncta-like objects are not independently confirmed telomere ends in this two-channel TRF2/DAPI dataset.",
        "",
        "Primary recommendation:",
    ]
    if best.empty:
        lines.append("- No condition passed the automated suitability screen.")
    else:
        for _, row in best.iterrows():
            lines.append(
                "- "
                f"{row['condition_label']}: n={int(row['n_valid_nuclei'])}, "
                f"mean background-subtracted TRF2={row['mean_background_subtracted_trf2']:.2f}, "
                f"mean puncta/nucleus={row['mean_puncta_per_nucleus']:.2f}."
            )
    if not review.empty:
        lines += ["", "Manual review candidates:"]
        for _, row in review.iterrows():
            lines.append(
                "- "
                f"{row['condition_label']}: {row['suitability_call']}; "
                f"mean puncta/nucleus={row['mean_puncta_per_nucleus']:.2f}."
            )
    lines += [
        "",
        "None-antibody wells are useful controls, not suitable staining conditions.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def cx5461_dilution_summary(labelled_nuclei: pd.DataFrame) -> pd.DataFrame:
    valid = labelled_nuclei[
        labelled_nuclei["valid_for_summary"].astype(str).str.upper().eq("TRUE")
        & labelled_nuclei["antibody_present"]
        & labelled_nuclei["drug"].eq("1 uM CX5461")
    ].copy()
    return (
        valid.groupby("trf2_dilution", sort=True)
        .agg(
            n_valid_nuclei=("nucleus_label", "count"),
            mean_background_subtracted_trf2=("nucleus_background_subtracted_mean_trf2", "mean"),
            median_background_subtracted_trf2=("nucleus_background_subtracted_mean_trf2", "median"),
            mean_puncta_per_nucleus=("trf2_puncta_count", "mean"),
            median_puncta_per_nucleus=("trf2_puncta_count", "median"),
        )
        .reset_index()
        .sort_values(["mean_background_subtracted_trf2", "n_valid_nuclei"], ascending=[False, False])
    )


def write_practical_recommendation(path: Path, cx_summary: pd.DataFrame) -> None:
    lines = [
        "# Practical TRF2Opt recommendation",
        "",
        "Best condition for further CX5461 studies: **TRF2 1:1000**.",
        "",
        "Reasoning:",
        "- Across CX5461 wells, 1:1000 gave the strongest mean background-subtracted nuclear TRF2 signal while retaining moderate puncta-like detection.",
        "- 1:2000 is a usable backup if antibody conservation or lower staining intensity is preferred.",
        "- 1:300 is usable but more concentrated than needed for the observed signal.",
        "- 1:500 was lower in nuclear TRF2 signal and is not the first choice.",
        "- None-antibody wells should be kept as negative controls, not used as staining conditions.",
        "",
        "CX5461 dilution summary:",
    ]
    for _, row in cx_summary.iterrows():
        lines.append(
            "- "
            f"TRF2 {row['trf2_dilution']}: n={int(row['n_valid_nuclei'])}, "
            f"mean background-subtracted TRF2={row['mean_background_subtracted_trf2']:.2f}, "
            f"mean puncta/nucleus={row['mean_puncta_per_nucleus']:.2f}."
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            parser.error(f"Refusing to overwrite existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    sample_sheet = load_map()
    sample_sheet.to_csv(output_dir / "sample_sheet_mapping.csv", index=False)
    (output_dir / "sample_sheet_mapping.json").write_text(
        json.dumps(SAMPLE_SHEET, indent=2) + "\n",
        encoding="utf-8",
    )

    labelled_frames: dict[str, pd.DataFrame] = {}
    for name in CSV_FILES:
        frame = pd.read_csv(input_dir / name)
        labelled = label_frame(frame, sample_sheet)
        labelled.to_csv(output_dir / name.replace(".csv", "_labelled.csv"), index=False)
        labelled_frames[name] = labelled

    suitability = suitability_table(labelled_frames["nucleus_measurements.csv"])
    suitability.to_csv(output_dir / "suitability_summary.csv", index=False)
    cx_summary = cx5461_dilution_summary(labelled_frames["nucleus_measurements.csv"])
    cx_summary.to_csv(output_dir / "cx5461_dilution_summary.csv", index=False)
    write_report(output_dir / "suitability_recommendation.md", suitability)
    write_practical_recommendation(output_dir / "practical_recommendation.md", cx_summary)
    print(f"Wrote labelled TRF2Opt outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
