#!/usr/bin/env python3
"""Generate a CZI analysis sample sheet from paths and filenames only."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


SERIES_ROOT = Path("/Volumes/Backup/TS runs")

PATTERNS = [
    re.compile(
        r"^(?P<timepoint>\d+)\s*hr\s+20x\s+(?P<condition>[A-H])-(?P<replicate>\d+)\.czi$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<timepoint>\d+)\s*hr\s+20x\s+(?P<condition>.+)-(?P<replicate>\d+)\.czi$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<timepoint>\d+)\s*hr\s+(?P<condition>.+?)\s+20x\s*-?\s*(?P<replicate>\d+)\.czi$",
        re.IGNORECASE,
    ),
]


FIELDNAMES = [
    "sample_id",
    "source_path",
    "relative_path",
    "file_name",
    "file_size_bytes",
    "time_series",
    "time_series_number",
    "time_series_note",
    "acquisition_date_folder",
    "timepoint_hr",
    "condition",
    "condition_type",
    "replicate",
    "magnification",
    "is_vehicle_or_control",
    "requires_manual_condition_mapping",
    "parse_status",
    "parse_note",
]


def split_series_name(series_name: str) -> tuple[str, str]:
    match = re.match(r"^Time Series\s+([IVX]+)(?:\s+\((.+)\))?$", series_name)
    if not match:
        return "", ""
    return match.group(1), match.group(2) or ""


def normalise_condition(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip())


def condition_type(condition: str) -> str:
    if re.fullmatch(r"[A-H]", condition):
        return "coded"
    if condition.lower() in {"aqueous", "dmso", "nah2po4"}:
        return "vehicle_or_control"
    return "named_treatment"


def parse_file(path: Path, root: Path) -> dict[str, str | int | bool]:
    relative = path.relative_to(root)
    parts = relative.parts
    root_series_number, _ = split_series_name(root.name)
    if root_series_number:
        series_name = root.name
        date_folder = parts[0] if len(parts) > 0 else ""
    else:
        series_name = parts[0] if len(parts) > 0 else ""
        date_folder = parts[1] if len(parts) > 1 else ""
    filename = path.name
    series_number, series_note = split_series_name(series_name)

    parsed = None
    for pattern in PATTERNS:
        parsed = pattern.match(filename)
        if parsed:
            break

    parse_status = "ok" if parsed else "unparsed"
    parse_note = ""
    timepoint_hr = ""
    condition = ""
    replicate = ""
    if parsed:
        timepoint_hr = parsed.group("timepoint")
        condition = normalise_condition(parsed.group("condition"))
        replicate = parsed.group("replicate").zfill(2)
    else:
        parse_note = "Filename did not match expected CZI naming patterns."

    ctype = condition_type(condition) if condition else ""
    requires_mapping = ctype == "coded"
    if requires_mapping:
        parse_note = "Condition is coded A-H; treatment mapping is not inferable from filename or directory."
    elif "weird controls" in series_name.lower():
        parse_note = "Series label contains 'weird controls'; review control interpretation before pooled analysis."

    sample_bits = [
        f"TS{series_number}" if series_number else "TSUNK",
        date_folder,
        f"{timepoint_hr}hr" if timepoint_hr else "timeUNK",
        re.sub(r"[^A-Za-z0-9]+", "_", condition).strip("_") if condition else "conditionUNK",
        f"rep{replicate}" if replicate else "repUNK",
    ]

    return {
        "sample_id": "_".join(sample_bits),
        "source_path": str(path),
        "relative_path": str(relative),
        "file_name": filename,
        "file_size_bytes": path.stat().st_size,
        "time_series": series_name,
        "time_series_number": series_number,
        "time_series_note": series_note,
        "acquisition_date_folder": date_folder,
        "timepoint_hr": timepoint_hr,
        "condition": condition,
        "condition_type": ctype,
        "replicate": replicate,
        "magnification": "20x" if "20x" in filename else "",
        "is_vehicle_or_control": "TRUE" if ctype == "vehicle_or_control" else "FALSE",
        "requires_manual_condition_mapping": "TRUE" if requires_mapping else "FALSE",
        "parse_status": parse_status,
        "parse_note": parse_note,
    }


def write_summary(rows: list[dict[str, str | int | bool]], path: Path) -> None:
    counts = Counter(
        (
            str(row["time_series_number"]),
            str(row["acquisition_date_folder"]),
            str(row["timepoint_hr"]),
            str(row["condition"]),
        )
        for row in rows
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["time_series_number", "acquisition_date_folder", "timepoint_hr", "condition", "n_files"])
        for key, count in sorted(counts.items()):
            writer.writerow([*key, count])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SERIES_ROOT)
    parser.add_argument("--output", type=Path, default=Path("sample_sheet.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("sample_sheet_summary.tsv"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"Root directory not found: {args.root}")
    for output in (args.output, args.summary_output):
        if output.exists() and not args.overwrite:
            parser.error(f"Refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    rows = [parse_file(path, args.root) for path in sorted(args.root.rglob("*.czi"))]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    write_summary(rows, args.summary_output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote summary to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
