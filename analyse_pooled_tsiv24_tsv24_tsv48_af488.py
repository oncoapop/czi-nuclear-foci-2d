#!/usr/bin/env python3
"""Field-level AF488 analysis pooling TS IV 24 h and TS V 24/48 h.

This script consumes existing field summaries only. It does not open CZI files
or rerun nuclei/focus segmentation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages


INPUT_CSV = Path("output/segmentation/batch_aware_current_dataset/field_batch_summary.csv")
OUTPUT_DIR = Path("output/segmentation/pooled_TSIV24_TSV24_TSV48_AF488_stats_v3_2026-07-29")
TREATMENTS = ["CX-5461", "Cisplatin", "Etoposide", "PDS", "Palbociclib"]
GROUP_ORDER = ["Control", *TREATMENTS]
STRATA = ["TSIV_24h", "TSV_24h", "TSV_48h"]
PANEL_LABELS = {
    "TSIV_24h": "Expt (1), 24 h",
    "TSV_24h": "Expt (2), 24 h",
    "TSV_48h": "Expt (3), 48 h",
}


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def fit_hc3_ols(design: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Return OLS coefficients, HC3 covariance, residual df, and residuals."""
    xtx_inverse = np.linalg.inv(design.T @ design)
    beta = xtx_inverse @ design.T @ response
    residuals = response - design @ beta
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    scaled_residuals = (residuals / (1.0 - leverage)) ** 2
    covariance = xtx_inverse @ (design.T @ (design * scaled_residuals[:, None])) @ xtx_inverse
    return beta, covariance, len(response) - design.shape[1], residuals


def build_design(data: pd.DataFrame, include_interactions: bool) -> tuple[np.ndarray, list[str], dict[tuple[str, str], int]]:
    columns = [np.ones(len(data))]
    names = ["Intercept (Control, TS IV 24 h)"]
    for stratum in STRATA[1:]:
        columns.append((data["stratum"] == stratum).astype(float).to_numpy())
        names.append(f"Stratum: {stratum}")
    for treatment in TREATMENTS:
        columns.append((data["analysis_group"] == treatment).astype(float).to_numpy())
        names.append(f"Treatment: {treatment} vs Control")
    interaction_columns: dict[tuple[str, str], int] = {}
    if include_interactions:
        for stratum in STRATA[1:]:
            for treatment in TREATMENTS:
                columns.append(((data["stratum"] == stratum) & (data["analysis_group"] == treatment)).astype(float).to_numpy())
                interaction_columns[(stratum, treatment)] = len(columns) - 1
                names.append(f"Interaction: {stratum} x {treatment}")
    return np.column_stack(columns), names, interaction_columns


def validate_input(data: pd.DataFrame) -> None:
    required = {
        "experiment", "timepoint_hr", "condition", "sample_id", "known_control",
        "n_nuclei", "mean_af488_spots_per_nucleus",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data["sample_id"].duplicated().any():
        raise ValueError("Sample IDs are not unique at field level")
    if data["mean_af488_spots_per_nucleus"].isna().any():
        raise ValueError("Missing AF488 field means")
    if (data["n_nuclei"] <= 0).any():
        raise ValueError("Zero-nucleus fields must be excluded or justified")
    expected = 96
    if len(data) != expected:
        raise ValueError(f"Expected {expected} selected fields, found {len(data)}")
    per_condition = data.groupby(["stratum", "condition"], dropna=False).size()
    if not (per_condition == 4).all():
        raise ValueError("Expected four fields per stratum/condition")


def write_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stratum, condition, known_control), group in data.groupby(["stratum", "condition", "known_control"], sort=False):
        values = group["mean_af488_spots_per_nucleus"]
        rows.append(
            {
                "stratum": stratum,
                "condition": condition,
                "known_control": bool(known_control),
                "n_fields": len(group),
                "total_nuclei": int(group["n_nuclei"].sum()),
                "mean_af488_spots_per_nucleus": float(values.mean()),
                "median_af488_spots_per_nucleus": float(values.median()),
                "sd_field_means": float(values.std(ddof=1)),
                "sem_field_means": float(values.sem(ddof=1)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "af488_field_level_summary.csv", index=False)
    return summary


def run_model(data: pd.DataFrame) -> pd.DataFrame:
    design, names, interaction_columns = build_design(data, include_interactions=True)
    response = data["log1p_af488_spots_per_nucleus"].to_numpy(dtype=float)
    beta, covariance, residual_df, residuals = fit_hc3_ols(design, response)
    standard_errors = np.sqrt(np.diag(covariance))
    t_values = beta / standard_errors
    p_values = 2.0 * stats.t.sf(np.abs(t_values), df=residual_df)
    critical_value = stats.t.ppf(0.975, df=residual_df)

    interaction_indices = list(interaction_columns.values())
    interaction_beta = beta[interaction_indices]
    interaction_covariance = covariance[np.ix_(interaction_indices, interaction_indices)]
    wald_statistic = float(interaction_beta @ np.linalg.inv(interaction_covariance) @ interaction_beta)
    interaction_df = len(interaction_indices)
    interaction_f = wald_statistic / interaction_df
    interaction_p = float(stats.f.sf(interaction_f, interaction_df, residual_df))
    pd.DataFrame(
        [{
            "test": "all treatment-by-stratum interactions equal zero",
            "wald_chi_square_approximation": wald_statistic,
            "f_statistic": interaction_f,
            "numerator_df": interaction_df,
            "denominator_df": residual_df,
            "p_value_hc3": interaction_p,
            "pooling_supported": interaction_p >= 0.05,
        }]
    ).to_csv(OUTPUT_DIR / "af488_treatment_by_stratum_interaction_hc3.csv", index=False)

    rows = []
    for stratum in STRATA:
        for treatment_index, treatment in enumerate(TREATMENTS, start=3):
            contrast = np.zeros(len(beta))
            contrast[treatment_index] = 1.0
            if stratum != STRATA[0]:
                contrast[interaction_columns[(stratum, treatment)]] = 1.0
            estimate = float(contrast @ beta)
            standard_error = float(np.sqrt(contrast @ covariance @ contrast))
            t_statistic = estimate / standard_error
            p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=residual_df))
            rows.append(
                {
                    "stratum": stratum,
                    "treatment": treatment,
                    "comparison": "treatment vs pooled mapped controls within stratum",
                    "n_treatment_fields": int(((data["stratum"] == stratum) & (data["analysis_group"] == treatment)).sum()),
                    "n_control_fields": int(((data["stratum"] == stratum) & (data["analysis_group"] == "Control")).sum()),
                    "log1p_effect": estimate,
                    "hc3_robust_se": standard_error,
                    "ci95_log1p_lower": float(estimate - critical_value * standard_error),
                    "ci95_log1p_upper": float(estimate + critical_value * standard_error),
                    "t_statistic": float(t_statistic),
                    "p_value_hc3": p_value,
                    "ratio_1_plus_af488_mean": float(np.exp(estimate)),
                    "residual_df": int(residual_df),
                }
            )
    contrasts = pd.DataFrame(rows)
    contrasts["p_value_bh_fdr_15_contrasts"] = benjamini_hochberg(contrasts["p_value_hc3"].to_numpy(dtype=float))
    contrasts.to_csv(OUTPUT_DIR / "af488_stratum_specific_treatment_vs_controls_hc3.csv", index=False)

    coefficient_table = pd.DataFrame(
        {
            "term": names,
            "coefficient_log1p": beta,
            "hc3_robust_se": standard_errors,
            "t_statistic": t_values,
            "p_value_hc3": p_values,
        }
    )
    coefficient_table.to_csv(OUTPUT_DIR / "af488_stratum_interaction_model_coefficients_hc3.csv", index=False)
    data.assign(model_residual_log1p=residuals).to_csv(OUTPUT_DIR / "af488_model_input_with_residuals.csv", index=False)
    return contrasts


def format_q_value(value: float) -> str:
    return "q<0.001" if value < 0.001 else f"q={value:.3f}"


def make_figure(data: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    rng = np.random.default_rng(20260729)
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.7})
    colours = {"Control": "#B7B7B7", "CX-5461": "#365F8C", "Cisplatin": "#365F8C", "Etoposide": "#365F8C", "PDS": "#365F8C", "Palbociclib": "#365F8C"}
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.25), sharey=True)
    y_limit = data["mean_af488_spots_per_nucleus"].max() * 1.24
    for axis, stratum in zip(axes, STRATA):
        subset = data[data["stratum"] == stratum]
        for index, group_name in enumerate(GROUP_ORDER):
            values = subset.loc[subset["analysis_group"] == group_name, "mean_af488_spots_per_nucleus"].to_numpy(dtype=float)
            x_values = np.full(values.shape, index, dtype=float) + rng.uniform(-0.10, 0.10, len(values))
            axis.scatter(x_values, values, s=18, facecolor=colours[group_name], edgecolor="#222222", linewidth=0.45, zorder=3)
            axis.errorbar(index, values.mean(), yerr=values.std(ddof=1) / np.sqrt(len(values)), fmt="_", color="#111111", markersize=10, elinewidth=0.8, capsize=0, zorder=4)
        panel_contrasts = contrasts[contrasts["stratum"] == stratum].set_index("treatment")
        for index, treatment in enumerate(TREATMENTS, start=1):
            axis.text(index, y_limit * 0.985, format_q_value(float(panel_contrasts.loc[treatment, "p_value_bh_fdr_15_contrasts"])), ha="center", va="top", fontsize=5.4, color="#222222")
        axis.set_title(PANEL_LABELS[stratum], loc="left", fontsize=8, fontweight="bold", pad=3)
        axis.set_xticks(range(len(GROUP_ORDER)))
        axis.set_xticklabels(["Control", "CX-5461", "Cisplatin", "Etoposide", "PDS", "Palbociclib"], rotation=45, ha="right")
        axis.set_xlim(-0.55, len(GROUP_ORDER) - 0.45)
        axis.set_ylim(0, y_limit)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.5)
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", width=0.7, length=3, labelsize=6.5)
    axes[0].set_ylabel("AF488 foci per nucleus\n(field mean)", fontsize=7)
    figure.text(0.01, 0.995, "AF488/53BP1 foci", ha="left", va="top", fontsize=8, fontweight="bold")
    figure.text(0.99, 0.995, "Dots: CZI fields; horizontal bar: mean +/- s.e.m.; q: FDR-adjusted treatment vs control", ha="right", va="top", fontsize=6, color="#4D4D4D")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    with PdfPages(OUTPUT_DIR / "AF488_field_level_statistics_nature_style.pdf") as pdf:
        pdf.savefig(figure)
    figure.savefig(OUTPUT_DIR / "AF488_field_level_statistics_nature_style.png", dpi=600)
    plt.close(figure)


def write_methods() -> None:
    (OUTPUT_DIR / "analysis_methods.txt").write_text(
        "Primary outcome: field-level mean AF488 spots per nucleus.\n\n"
        "Included fields: TS IV 24 h, TS V 24 h, and TS V 48 h only.\n"
        "Unit of analysis: one CZI field. Individual nuclei were not treated as independent replicates.\n\n"
        "Controls: all rows where known_control is True in the validated field summary.\n"
        "Model: ordinary least squares on log(1 + field mean AF488 spots per nucleus), with fixed effects for experiment/timepoint stratum, treatment group, and their interaction.\n"
        "Strata: TS IV 24 h, TS V 24 h, TS V 48 h.\n"
        "Pooling check: a global HC3 robust Wald test evaluates whether all treatment-by-stratum interactions are zero.\n"
        "Primary contrasts: treatment versus pooled mapped controls within each stratum, using HC3 heteroscedasticity-consistent standard errors and two-sided t tests.\n"
        "Multiplicity: Benjamini-Hochberg correction across all 15 stratum-specific treatment-versus-control contrasts.\n\n"
        "Interpretation limit: four fields per condition/stratum are imaging fields, not confirmed biological replicates. P values therefore support exploratory field-level comparisons only and should not be used as biological-replicate inference.\n",
        encoding="utf-8",
    )


def main() -> None:
    source = pd.read_csv(INPUT_CSV)
    selected = source[
        ((source["experiment"] == "TS IV") & (source["timepoint_hr"] == 24))
        | ((source["experiment"] == "TS V") & (source["timepoint_hr"].isin([24, 48])))
    ].copy()
    selected["stratum"] = selected["experiment"].str.replace(" ", "", regex=False) + "_" + selected["timepoint_hr"].astype(str) + "h"
    selected["analysis_group"] = np.where(selected["known_control"], "Control", selected["condition"])
    selected["log1p_af488_spots_per_nucleus"] = np.log1p(selected["mean_af488_spots_per_nucleus"])
    validate_input(selected)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    selected.to_csv(OUTPUT_DIR / "selected_field_level_input.csv", index=False)
    write_summary(selected)
    contrasts = run_model(selected)
    make_figure(selected, contrasts)
    write_methods()
    print(f"Wrote pooled AF488 analysis for {len(selected)} fields to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
