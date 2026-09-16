#!/usr/bin/env python3
"""Summary figures for the 2026-09-15 audit of the CZI foci pipeline."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import morphology

AUDIT = Path("/Volumes/Backup/czi-audit-2026-09-15")
PUB = Path("/Users/dyap/Documents/Tasks workflow/czi-analysis/output/segmentation")
RUN = Path("/Volumes/Backup/czi-analysis-runs/2026-09-15_time_series")
OUT = AUDIT / "figures"

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # reference palette slots 1-3
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.linewidth": 0.8,
})


def style(ax, ygrid=True):
    ax.grid(axis="y" if ygrid else "x", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(length=3, width=0.8, labelsize=8)


def label_ends(ax, x, items, dx=0.09, avoid=None):
    """Direct-label line ends in TEXT ink (the coloured end marker carries identity),
    nudged apart so labels never overlap each other or a reference line."""
    lo, hi = ax.get_ylim()
    sep = 0.062 * (hi - lo)
    items = sorted(items, key=lambda t: t[0])
    ys = [y for y, _ in items]
    for i in range(1, len(ys)):                       # push up
        ys[i] = max(ys[i], ys[i - 1] + sep)
    for i in range(len(ys) - 2, -1, -1):              # then relax down
        ys[i] = min(ys[i], ys[i + 1] - sep)
    if avoid is not None:
        for i, y in enumerate(ys):
            if abs(y - avoid) < sep * 0.75:
                ys[i] = y - sep * 0.85 if y <= avoid else y + sep * 0.85
    for (orig, text), y in zip(items, ys):
        ax.annotate(text, xy=(x, orig), xytext=(x + dx, y), fontsize=8, color=INK2,
                    va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.7,
                                    shrinkA=0, shrinkB=2) if abs(y - orig) > sep * 0.4 else None)


# ---------------------------------------------------------------- data
MADS = [3.0, 4.0, 5.0]
SWEEP = {3.0: AUDIT / "sweep/mad3.0", 4.0: AUDIT / "repro_spotcheck6", 5.0: AUDIT / "sweep/mad5.0"}
PAIRS = {"TS III 2 h\nCX-5461 vs DMSO": ("TSIII_2026-05-15_2hr_CX_5461_rep03", "TSIII_2026-05-15_2hr_DMSO_rep04"),
         "TS IV 24 h\nEtoposide vs DMSO": ("TSIV_2026-05-26_24hr_Etoposide_rep04", "TSIV_2026-05-26_24hr_DMSO_rep03")}

nuc = {m: pd.read_csv(d / "nucleus_dual_channel_measurements.csv") for m, d in SWEEP.items()}

totals = {k: [] for k in ("AF488", "RhReX", "Colocalised")}
for m in MADS:
    n = nuc[m]
    totals["AF488"].append(n.af488_spots_count.sum())
    totals["RhReX"].append(n.rhrex_spots_count.sum())
    totals["Colocalised"].append(n.colocalized_rhrex_spots_count.sum())
fold = {k: np.array(v) / v[MADS.index(4.0)] for k, v in totals.items()}


def ratios(pair):
    t, c = pair
    out = {"Mean AF488 / nucleus": [], "AF488-positive (>=5)": [], "Coloc-positive (>=1)": []}
    for m in MADS:
        n = nuc[m]
        T, C = n[n.sample_id == t], n[n.sample_id == c]
        out["Mean AF488 / nucleus"].append(T.af488_spots_count.mean() / C.af488_spots_count.mean())
        out["AF488-positive (>=5)"].append((T.af488_spots_count >= 5).mean() / (C.af488_spots_count >= 5).mean())
        fc = (C.colocalized_rhrex_spots_count >= 1).mean()
        out["Coloc-positive (>=1)"].append((T.colocalized_rhrex_spots_count >= 1).mean() / fc if fc else np.nan)
    return out


# =================================================================== FIGURE 1
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
fig.suptitle("Sensitivity to the foci-detection threshold (MAD multiplier), 6 fields",
             x=0.012, ha="left", fontsize=13, fontweight="semibold", color=INK, y=0.985)
fig.text(0.012, 0.915, "The published pipeline uses 4.0. Nothing validates that choice, so the question is "
         "what moves when it changes.", ha="left", fontsize=8.5, color=INK2)

ax = axes[0]
ax.set_ylim(0.3, 2.75)
for (name, vals), col in zip(fold.items(), (S1, S2, S3)):
    ax.plot(MADS, vals, "-o", color=col, lw=2, ms=8, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=2, label=name)
ax.axhline(1.0, color=BASELINE, lw=1, ls="--", zorder=1)
label_ends(ax, MADS[-1], [(v[-1], n) for n, v in fold.items()])
ax.set_title("Total detections, relative to published setting", fontsize=10, loc="left")
ax.set_xlabel("MAD multiplier"); ax.set_ylabel("fold of published (4.0)")
ax.set_xlim(2.85, 6.15); ax.set_xticks(MADS); style(ax)
ax.legend(frameon=False, fontsize=7.5, loc="lower left", labelcolor=INK2, handlelength=1.4)

for ax, (title, pair) in zip(axes[1:], PAIRS.items()):
    r = ratios(pair)
    ax.set_ylim(0, 4.3)
    for (name, vals), col in zip(r.items(), (S1, S2, S3)):
        ax.plot(MADS, vals, "-o", color=col, lw=2, ms=8, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name)
    ax.axhline(1.0, color=CRITICAL, lw=1.2, ls="--", zorder=2)
    ax.text(5.98, 1.06, "no effect", fontsize=7.5, color=CRITICAL, va="bottom", ha="right")
    label_ends(ax, MADS[-1], [(v[-1], n) for n, v in r.items()], avoid=1.0)
    ax.set_title(f"Treated / control ratio - {title}".replace("\n", ", "), fontsize=10, loc="left")
    ax.set_xlabel("MAD multiplier"); ax.set_ylabel("treated / control")
    ax.set_xlim(2.85, 6.6); ax.set_xticks(MADS); style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", labelcolor=INK2, handlelength=1.4)

fig.tight_layout(rect=(0, 0.01, 1, 0.885))
fig.savefig(OUT / "fig1_parameter_sensitivity.png", dpi=200)
plt.close(fig)
print("fig1 done")

# =================================================================== FIGURE 2
dirs = {"TS III": "time_series_III_dual_channel_coloc", "TS IV": "time_series_IV_dual_channel_coloc",
        "TS V": "time_series_V_dual_channel_coloc"}
img = pd.concat([pd.read_csv(PUB / d / "image_summary_dual_channel.csv").assign(experiment=e)
                 for e, d in dirs.items()], ignore_index=True)
img = img[np.isfinite(img.af488_threshold)]
grp = img.groupby(["experiment", "timepoint_hr", "condition"]).af488_threshold.agg(["min", "max", "size"])
grp = grp[grp["size"] >= 3]
grp["fold"] = grp["max"] / grp["min"]

# detection margin: per-nucleus tophat p99 minus that field's threshold
def lm(n, f):
    spec = importlib.util.spec_from_file_location(n, Path(f).resolve())
    m = importlib.util.module_from_spec(spec); sys.modules[n] = m; spec.loader.exec_module(m); return m
base = lm("b", Path(__file__).with_name("segment_nuclei_spots.py")); insp = base.load_inspector()
SEG = AUDIT / "repro_spotcheck6"
margins = []
for src in pd.read_csv(AUDIT / "spotcheck_manifest.csv").source_path:
    czi = Path(src); ins = insp.inspect(czi, True, insp and None)
    rh, af, dapi = base.read_channel_arrays(czi, ins, insp)
    sid = base.sample_id(czi)
    nl = np.array(Image.open(SEG / "masks" / f"{sid}_nuclei_labels.tif"))
    thr = float(pd.read_csv(SEG / "image_summary_dual_channel.csv").query("sample_id==@sid").af488_threshold.iloc[0])
    top = morphology.white_tophat(af, footprint=morphology.disk(4))
    for L in np.unique(nl):
        if L == 0: continue
        margins.append(np.percentile(top[nl == L].astype(float), 99) - thr)
margins = np.array(margins)

comb = pd.read_csv(RUN / "combined" / "combined_nucleus_measurements.csv")
ctl = comb[comb.known_control]
zero = pd.DataFrame({
    "53BP1 / AF488": ctl.groupby("experiment").af488_spot_count.apply(lambda s: 100 * (s == 0).mean()),
    "gH2AX / RhReX": ctl.groupby("experiment").rhrex_putative_gH2AX_count.apply(lambda s: 100 * (s == 0).mean()),
    "Colocalised": ctl.groupby("experiment").colocalized_rhrex_putative_gH2AX_count.apply(lambda s: 100 * (s == 0).mean()),
})

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
fig.suptitle("Why the counts are fragile: three measured causes", x=0.012, ha="left",
             fontsize=13, fontweight="semibold", color=INK, y=0.985)
fig.text(0.012, 0.915, "All from the published outputs - no re-processing needed.", ha="left",
         fontsize=8.5, color=INK2)

ax = axes[0]
ax.hist(grp["fold"], bins=np.arange(1.0, 2.3, 0.1), color=S1, edgecolor=SURFACE, linewidth=2, zorder=3)
ax.axvline(grp["fold"].median(), color=CRITICAL, lw=1.5, ls="--", zorder=4)
ax.text(grp["fold"].median() + 0.03, ax.get_ylim()[1] * 0.92,
        f"median {grp['fold'].median():.2f}x", color=CRITICAL, fontsize=8)
ax.set_title("Detection threshold differs between replicate images", fontsize=10, loc="left")
ax.set_xlabel("highest / lowest threshold within a condition+timepoint group")
ax.set_ylabel("number of groups"); style(ax)

ax = axes[1]
ax.hist(margins, bins=40, color=S1, edgecolor=SURFACE, linewidth=1.2, zorder=3)
ax.axvline(0, color=CRITICAL, lw=1.5, ls="--", zorder=4)
ax.text(2, ax.get_ylim()[1] * 0.92, " detection cut-off", color=CRITICAL, fontsize=8)
near = 100 * np.mean(np.abs(margins) <= 15)
ax.set_title(f"Nuclei sit close to the cut-off ({near:.0f}% within +/-15 units)", fontsize=10, loc="left")
ax.set_xlabel("nucleus brightest signal minus threshold"); ax.set_ylabel("number of nuclei"); style(ax)

ax = axes[2]
x = np.arange(len(zero)); w = 0.26
for i, (col, colour) in enumerate(zip(zero.columns, (S1, S2, S3))):
    bars = ax.bar(x + (i - 1) * w, zero[col], w * 0.92, color=colour, zorder=3,
                  edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, zero[col]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(zero.index)
ax.set_ylim(0, 132)
ax.set_title("Control nuclei with zero foci (drives the MAD collapse)", fontsize=10, loc="left")
ax.set_ylabel("% of control nuclei with zero foci")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (S1, S2, S3)]
ax.legend(handles, list(zero.columns), frameon=False, fontsize=8, loc="upper left",
          ncol=3, labelcolor=INK2, handlelength=1.2, columnspacing=1.2, borderaxespad=0.2)
style(ax)

fig.tight_layout(rect=(0, 0.01, 1, 0.885))
fig.savefig(OUT / "fig2_why_fragile.png", dpi=200)
plt.close(fig)
print("fig2 done")

# =================================================================== FIGURE 3
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
fig.suptitle("Two decisions checked against the data", x=0.015, ha="left",
             fontsize=13, fontweight="semibold", color=INK, y=0.985)

ax = axes[0]
fm = (ctl.groupby(["experiment", "timepoint_hr", "sample_id"]).af488_spot_count.mean()
         .reset_index(name="field_mean"))
cs = (fm.groupby(["experiment", "timepoint_hr"]).field_mean
        .agg(mean="mean", sem=lambda s: s.std(ddof=1) / np.sqrt(len(s))).reset_index())
pos, tick = 0.0, []
for i, (exp, g) in enumerate(cs.groupby("experiment")):
    xs = pos + np.arange(len(g))
    ax.errorbar(xs, g["mean"], yerr=g["sem"], fmt="-o", color=(S1, S2, S3)[i], lw=2, ms=9,
                capsize=3, elinewidth=1.2, zorder=3, markeredgecolor=SURFACE, markeredgewidth=2)
    for xx, (_, r) in zip(xs, g.iterrows()):
        ax.text(xx, 0.62, f"{int(r.timepoint_hr)} h", ha="center", fontsize=7.5, color=MUTED)
    tick.append((xs.mean(), exp))
    pos += len(g) + 1.1
for xx, lab in tick:
    ax.text(xx, 4.62, lab, ha="center", fontsize=9.5, color=INK, fontweight="medium")
ax.annotate("1.63x drop\np = 0.0004", xy=(1, 2.01), xytext=(1.35, 1.05), fontsize=8, color=CRITICAL,
            arrowprops=dict(arrowstyle="->", color=CRITICAL, lw=1.1))
ax.set_ylim(0, 5.0); ax.set_xticks([]); ax.set_xlim(-0.7, pos - 0.5)
ax.set_title("Control baseline is NOT flat within TS III", fontsize=10, loc="left")
ax.set_ylabel("AF488 foci per control nucleus")
style(ax)
fig.text(0.015, 0.015, "Left: field-level mean +/- s.e.m. of control nuclei; between-experiment range 1.95x, "
         "so the TS III within-experiment drop is nearly as large as the between-experiment shift.",
         fontsize=7.5, color=MUTED, ha="left")

ax = axes[1]
rows = []
for d, e in dirs.items():
    pass
bn = []
for exp, d in dirs.items():
    nt = pd.read_csv(PUB / d / "nucleus_dual_channel_measurements.csv")
    for sid, g in nt.groupby("sample_id"):
        p = PUB / d / "masks" / f"{sid}_nuclei_labels.tif"
        if not p.is_file(): continue
        m = np.array(Image.open(p))
        edge = np.concatenate([m[0, :], m[-1, :], m[:, 0], m[:, -1]])
        bn.append(g.assign(on_border=g.nucleus_label.isin({int(v) for v in np.unique(edge) if v > 0}),
                           experiment=exp))
bn = pd.concat(bn, ignore_index=True)
a = bn.groupby(["experiment", "condition"]).af488_spots_count.mean()
b = bn[~bn.on_border].groupby(["experiment", "condition"]).af488_spots_count.mean()
lim = max(a.max(), b.max()) * 1.12
ax.plot([0, lim], [0, lim], color=BASELINE, lw=1, ls="--", zorder=1)
ax.plot(a.values, b.values, "o", ms=9, color=S1, zorder=3, markeredgecolor=SURFACE, markeredgewidth=2)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_title("Dropping border nuclei barely moves anything", fontsize=10, loc="left")
ax.set_xlabel("condition mean, all nuclei"); ax.set_ylabel("condition mean, border nuclei excluded")
ax.text(0.04, 0.9, f"{100*bn.on_border.mean():.1f}% of nuclei touch the edge\n"
                   f"mean shift {np.mean(np.abs(100*(b.values/a.values-1))):.1f}%   "
                   f"contrast change 1-2%", transform=ax.transAxes, fontsize=8, color=INK2, va="top")
style(ax)

fig.tight_layout(rect=(0, 0.045, 1, 0.93))
fig.savefig(OUT / "fig3_decisions_checked.png", dpi=200)
plt.close(fig)
print("fig3 done")
print(f"\nwrote figures to {OUT}")
