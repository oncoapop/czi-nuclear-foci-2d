from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "output/publication_53BP1_v1/data"
POSITIVE_FIELDS = ROOT / "output/publication_53BP1_gt5_v1/tables/field_level_53BP1_summary.csv"
OUT = ROOT / "output/qc_nuclei_gt5_53BP1_per_field_pooled_v1"
TSV_MAP = {
    "A": "CX-5461",
    "B": "Cisplatin",
    "C": "Etoposide",
    "D": "Palbociclib",
    "E": "NAH2PO4",
    "F": "Media",
    "G": "DMSO",
    "H": "PDS",
}
TSV_DOSES = {
    "CX-5461": "1 µM",
    "Cisplatin": "1.2 µM",
    "Etoposide": "1 µM",
    "Palbociclib": "50 nM",
    "NAH2PO4": "50 µM",
    "PDS": "5 µM",
    "DMSO": "0.001%",
    "Media": "control",
}
CONTROLS = {"Aqueous", "Media", "DMSO"}
CONDITION_ORDER = ["Aqueous", "Media", "DMSO", "CX-5461", "Cisplatin", "Etoposide", "Palbociclib", "NAH2PO4", "PDS"]
EXPERIMENT_ORDER = ["TSIII", "TSIV", "TSV"]
MARKERS = {"TSIII": "o", "TSIV": "s", "TSV": "^"}


def dose_for(experiment: str, condition: str) -> str:
    if condition == "DMSO":
        return "0.001%"
    if condition in {"Aqueous", "Media"}:
        return "control"
    if experiment in {"TSIII", "TSIV"}:
        return "1 µM"
    return TSV_DOSES[condition]


def load_fields() -> pd.DataFrame:
    frames = []
    for experiment in EXPERIMENT_ORDER:
        frame = pd.read_csv(DATA / experiment / "image_summary.csv")
        frame["experiment"] = experiment
        if experiment == "TSV":
            frame["condition"] = frame["condition_code"].map(TSV_MAP)
        frames.append(frame)
    fields = pd.concat(frames, ignore_index=True)
    positive = pd.read_csv(POSITIVE_FIELDS)[["experiment", "sample_id", "nuclei_gt5_53BP1"]]
    fields = fields.merge(positive, on=["experiment", "sample_id"], how="left", validate="one_to_one")
    missing_positive = fields["nuclei_gt5_53BP1"].isna()
    if int(missing_positive.sum()) != 1 or not fields.loc[missing_positive, "nuclei_count"].eq(0).all():
        raise RuntimeError("Unexpected missing per-field >5 53BP1 count")
    fields["nuclei_gt5_53BP1"] = fields["nuclei_gt5_53BP1"].fillna(0).astype(int)
    fields["dose"] = [dose_for(e, c) for e, c in zip(fields["experiment"], fields["condition"])]
    fields["is_control"] = fields["condition"].isin(CONTROLS)
    fields["group_key"] = [
        f"{c}|{d}|{int(t)}h|{e if control else 'pooled'}"
        for e, c, d, t, control in zip(fields["experiment"], fields["condition"], fields["dose"], fields["timepoint_hr"], fields["is_control"])
    ]
    return fields


def build_group_table(fields: pd.DataFrame) -> pd.DataFrame:
    group = (
        fields.groupby(["group_key", "condition", "dose", "timepoint_hr", "is_control"], as_index=False)
        .agg(
            n_biological_replicates=("experiment", "nunique"),
            biological_replicates=("experiment", lambda x: "+".join(sorted(set(x), key=EXPERIMENT_ORDER.index))),
            n_fields=("sample_id", "count"),
            total_positive_nuclei=("nuclei_gt5_53BP1", "sum"),
            median_positive_nuclei_per_field=("nuclei_gt5_53BP1", "median"),
            q1_positive_nuclei_per_field=("nuclei_gt5_53BP1", lambda x: x.quantile(0.25)),
            q3_positive_nuclei_per_field=("nuclei_gt5_53BP1", lambda x: x.quantile(0.75)),
            min_positive_nuclei_per_field=("nuclei_gt5_53BP1", "min"),
            max_positive_nuclei_per_field=("nuclei_gt5_53BP1", "max"),
        )
    )
    condition_rank = {name: i for i, name in enumerate(CONDITION_ORDER)}
    group["condition_order"] = group["condition"].map(condition_rank)
    group["control_order"] = (~group["is_control"]).astype(int)
    group["experiment_order"] = group["biological_replicates"].map(lambda x: min(EXPERIMENT_ORDER.index(e) for e in x.split("+")))

    def label(row: pd.Series) -> str:
        name = {"NAH2PO4": "NaH2PO4"}.get(row["condition"], row["condition"])
        if row["condition"] == "DMSO":
            name += f" {row['dose']}"
        elif not row["is_control"]:
            name += f" {row['dose']}"
        if row["is_control"]:
            batch = row["biological_replicates"].replace("TSIII", "TS III").replace("TSIV", "TS IV").replace("TSV", "TS V")
            name += f" — {batch}"
        return f"{name}  (bio n={int(row['n_biological_replicates'])})"

    group["display_label"] = group.apply(label, axis=1)
    return group.sort_values(["timepoint_hr", "control_order", "condition_order", "dose", "experiment_order"]).reset_index(drop=True)


def make_plot(fields: pd.DataFrame, groups: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
    panels = [(2, "A"), (6, "B"), (24, "C"), (48, "D")]
    max_nuclei = int(fields["nuclei_gt5_53BP1"].max())
    x_max = int(np.ceil(max_nuclei / 25.0) * 25)

    for ax, (timepoint, letter) in zip(axes.flat, panels):
        panel_groups = groups.loc[groups["timepoint_hr"].eq(timepoint)].reset_index(drop=True)
        for row_index, group in panel_groups.iterrows():
            y = len(panel_groups) - 1 - row_index
            points = fields.loc[fields["group_key"].eq(group["group_key"])].copy()
            experiments = sorted(points["experiment"].unique(), key=EXPERIMENT_ORDER.index)
            offsets = {exp: (0.12 if len(experiments) == 2 and i == 0 else -0.12 if len(experiments) == 2 else 0.0) for i, exp in enumerate(experiments)}
            for experiment in experiments:
                exp_points = points.loc[points["experiment"].eq(experiment)].sort_values("replicate")
                jitter = np.linspace(-0.055, 0.055, len(exp_points)) if len(exp_points) > 1 else np.array([0.0])
                ax.scatter(
                    exp_points["nuclei_gt5_53BP1"],
                    y + offsets[experiment] + jitter,
                    marker=MARKERS[experiment],
                    s=26,
                    facecolors="white",
                    edgecolors="0.35",
                    linewidths=0.8,
                    zorder=3,
                )
                ax.scatter(
                    [exp_points["nuclei_gt5_53BP1"].median()],
                    [y + offsets[experiment]],
                    marker=MARKERS[experiment],
                    s=48,
                    facecolors="0.15",
                    edgecolors="0.15",
                    linewidths=0.8,
                    zorder=4,
                )
            median = float(group["median_positive_nuclei_per_field"])
            ax.plot([median, median], [y - 0.23, y + 0.23], color="0.0", linewidth=1.2, zorder=5)

        ax.set_yticks(range(len(panel_groups)))
        ax.set_yticklabels(panel_groups["display_label"].iloc[::-1])
        ax.set_xlim(-3, x_max)
        ax.set_ylim(-0.6, len(panel_groups) - 0.4)
        ax.grid(axis="x", color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"{letter}   {timepoint} h", loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("Nuclei with >5 53BP1 foci per field")

    handles = [
        plt.Line2D([], [], marker=MARKERS[e], linestyle="none", markerfacecolor="0.15", markeredgecolor="0.15", markersize=6, label=e.replace("TSIII", "TS III").replace("TSIV", "TS IV").replace("TSV", "TS V"))
        for e in EXPERIMENT_ORDER
    ]
    handles.append(plt.Line2D([], [], color="0.0", linewidth=1.2, label="Pooled field median"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("53BP1 focus QC: nuclei with >5 foci per imaging field", x=0.06, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.06, 0.945, "Open symbols: individual fields; filled symbols: biological-experiment median. Matching drug/dose/time groups are pooled; controls remain batch-specific.", ha="left", fontsize=9)
    fig.tight_layout(rect=(0.03, 0.055, 0.99, 0.925), h_pad=2.3, w_pad=4.0)
    fig.savefig(OUT / "QC_nuclei_gt5_53BP1_per_field_pooled.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "QC_nuclei_gt5_53BP1_per_field_pooled.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fields = load_fields()
    groups = build_group_table(fields)
    fields = fields.merge(groups[["group_key", "display_label", "n_biological_replicates"]], on="group_key", how="left")
    fields.to_csv(OUT / "field_level_nuclei_gt5_53BP1_counts.csv", index=False)
    groups.to_csv(OUT / "pooled_group_nuclei_gt5_53BP1_summary.csv", index=False)
    make_plot(fields, groups)
    print(f"Wrote {len(fields)} fields across {len(groups)} groups to {OUT}")
    print(f"Zero-positive fields: {int(fields['nuclei_gt5_53BP1'].eq(0).sum())}")
    print(f"Zero-nucleus fields retained: {int(fields['nuclei_count'].eq(0).sum())}")


if __name__ == "__main__":
    main()
