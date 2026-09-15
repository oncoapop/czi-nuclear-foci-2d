#!/usr/bin/env python3
"""Render per-nucleus crops for blind by-eye verification of foci detection.

Differs from make_coloc_zoom_examples.py, which shows only the single highest
co-localisation nucleus per field (a best case). This selects a stratified,
seeded sample across the algorithm's own AF488 count - including zero-count
nuclei and border-touching nuclei - so the sample is not chosen to flatter the
detector.

Emits two sheets per field:
  *_BLIND.png   raw channels only, no detection overlays  -> count these first
  *_ANSWER.png  same crops with detected foci outlined
plus a CSV of the algorithm's counts for the selected nuclei.

The nucleus boundary is drawn on every panel because the algorithm only counts a
focus for the nucleus containing the majority of its pixels; without it the
counting region is ambiguous.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import segmentation


def load_module(name: str, file_name: str):
    script = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stretch(image: np.ndarray) -> np.ndarray:
    """Per-crop contrast stretch, so dim foci are visible rather than crushed."""
    lo, hi = np.percentile(image, [1, 99.5])
    if hi <= lo:
        lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    return np.clip((image.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def to_rgb(gray: np.ndarray, outlines: list[tuple[np.ndarray, tuple[int, int, int]]]) -> np.ndarray:
    rgb = np.repeat(stretch(gray)[..., None], 3, axis=2)
    for labels, colour in outlines:
        if labels.max() > 0:
            rgb[segmentation.find_boundaries(labels, mode="outer")] = colour
    return rgb


def load_masks(seg_dir: Path, sid: str) -> dict[str, np.ndarray]:
    names = {
        "nuclei": f"{sid}_nuclei_labels.tif",
        "af488": f"{sid}_af488_spot_labels.tif",
        "rhrex": f"{sid}_rhrex_putative_gH2AX_labels.tif",
        "coloc": f"{sid}_colocalized_rhrex_labels.tif",
    }
    return {k: np.array(Image.open(seg_dir / "masks" / v)) for k, v in names.items()}


def border_labels(nuclei: np.ndarray) -> set[int]:
    edges = np.concatenate([nuclei[0, :], nuclei[-1, :], nuclei[:, 0], nuclei[:, -1]])
    return {int(v) for v in np.unique(edges) if v > 0}


def select_nuclei(rows: pd.DataFrame, on_border: set[int], n: int, rng) -> pd.DataFrame:
    """Stratify by AF488 count so the sample spans the detector's own range."""
    rows = rows.copy()
    rows["on_border"] = rows["nucleus_label"].isin(on_border)
    counts = rows["af488_spots_count"]

    zero = rows[counts == 0]
    nonzero = rows[counts > 0]
    if len(nonzero):
        q33, q66 = nonzero["af488_spots_count"].quantile([0.33, 0.66])
        low = nonzero[nonzero["af488_spots_count"] <= q33]
        mid = nonzero[(nonzero["af488_spots_count"] > q33) & (nonzero["af488_spots_count"] <= q66)]
        high = nonzero[nonzero["af488_spots_count"] > q66]
    else:
        low = mid = high = nonzero

    picked: list[pd.Series] = []
    seen: set[int] = set()

    border_pool = rows[rows["on_border"]]
    if len(border_pool):
        r = border_pool.sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
        picked.append(r); seen.add(int(r["nucleus_label"]))

    strata = [("zero", zero), ("low", low), ("mid", mid), ("high", high)]
    while len(picked) < n:
        progressed = False
        for _, pool in strata:
            if len(picked) >= n:
                break
            pool = pool[~pool["nucleus_label"].isin(seen)]
            if not len(pool):
                continue
            r = pool.sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
            picked.append(r); seen.add(int(r["nucleus_label"]))
            progressed = True
        if not progressed:
            break

    out = pd.DataFrame(picked)
    return out.sort_values("af488_spots_count").reset_index(drop=True)


def render(sid, czi, seg_dir, chosen, masks, arrays, out_dir, pad):
    rhrex, af488, dapi = arrays
    blind_cols = ["DAPI\n(yellow = nucleus)", "AF488 raw", "RhReX raw"]
    answer_cols = ["DAPI\n(yellow = nucleus)", "AF488 raw", "AF488 + detected\n(cyan)",
                   "RhReX raw", "RhReX + detected\n(orange)", "Co-localised\n(magenta)"]

    for mode, cols in (("BLIND", blind_cols), ("ANSWER", answer_cols)):
        nrow = len(chosen)
        fig, axes = plt.subplots(nrow, len(cols), figsize=(2.9 * len(cols), 3.0 * nrow), squeeze=False)
        for r, (_, row) in enumerate(chosen.iterrows()):
            lab = int(row["nucleus_label"])
            ys, xs = np.where(masks["nuclei"] == lab)
            y0, y1 = max(0, ys.min() - pad), min(dapi.shape[0], ys.max() + pad + 1)
            x0, x1 = max(0, xs.min() - pad), min(dapi.shape[1], xs.max() + pad + 1)
            ysl, xsl = slice(y0, y1), slice(x0, x1)

            this_nuc = np.where(masks["nuclei"][ysl, xsl] == lab, 1, 0)
            faint = (120, 120, 0)
            panels = {
                "DAPI\n(yellow = nucleus)": to_rgb(dapi[ysl, xsl], [(this_nuc, (255, 255, 0))]),
                "AF488 raw": to_rgb(af488[ysl, xsl], [(this_nuc, faint)]),
                "RhReX raw": to_rgb(rhrex[ysl, xsl], [(this_nuc, faint)]),
                "AF488 + detected\n(cyan)": to_rgb(af488[ysl, xsl],
                    [(this_nuc, faint), (masks["af488"][ysl, xsl], (0, 255, 255))]),
                "RhReX + detected\n(orange)": to_rgb(rhrex[ysl, xsl],
                    [(this_nuc, faint), (masks["rhrex"][ysl, xsl], (255, 128, 0))]),
                "Co-localised\n(magenta)": to_rgb(rhrex[ysl, xsl],
                    [(this_nuc, faint), (masks["coloc"][ysl, xsl], (255, 0, 255))]),
            }
            for c, title in enumerate(cols):
                ax = axes[r][c]
                ax.imshow(panels[title], interpolation="nearest")
                ax.set_axis_off()
                if r == 0:
                    ax.set_title(title, fontsize=9)
            tag = "  [BORDER]" if row.get("on_border") else ""
            axes[r][0].text(0, -6, f"nucleus {lab}{tag}", fontsize=9, va="bottom")

        title = f"{sid}   ({mode})"
        if mode == "BLIND":
            title += "   -- count foci inside the nucleus outline before viewing ANSWER"
        fig.suptitle(title, y=0.997, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        path = out_dir / f"{sid}_{mode}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  wrote {path.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segmentation-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True, help="CSV with source_path column")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--nuclei-per-field", type=int, default=8)
    ap.add_argument("--pad", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = load_module("segment_base_for_crops", "segment_nuclei_spots.py")
    inspector = base.load_inspector()
    nuclei_tbl = pd.read_csv(args.segmentation_dir / "nucleus_dual_channel_measurements.csv")
    rng = np.random.default_rng(args.seed)

    records = []
    for src in pd.read_csv(args.manifest)["source_path"]:
        czi = Path(src)
        inspection = inspector.inspect(czi, True, None)
        arrays = base.read_channel_arrays(czi, inspection, inspector)
        sid = base.sample_id(czi)
        masks = load_masks(args.segmentation_dir, sid)
        rows = nuclei_tbl[nuclei_tbl["sample_id"] == sid]
        if not len(rows):
            print(f"  !! no nucleus rows for {sid}"); continue
        chosen = select_nuclei(rows, border_labels(masks["nuclei"]), args.nuclei_per_field, rng)
        print(f"{sid}: {len(chosen)} nuclei selected of {len(rows)}")
        render(sid, czi, args.segmentation_dir, chosen, masks, arrays, args.output_dir, args.pad)
        for _, r in chosen.iterrows():
            records.append({
                "sample_id": sid, "nucleus_label": int(r["nucleus_label"]),
                "on_border": bool(r["on_border"]),
                "algo_af488_count": int(r["af488_spots_count"]),
                "algo_rhrex_count": int(r["rhrex_spots_count"]),
                "algo_coloc_count": int(r["colocalized_rhrex_spots_count"]),
                "nucleus_area_px": int(r["nucleus_area_px"]),
                "eye_af488_count": "", "eye_rhrex_count": "", "notes": "",
            })
    out_csv = args.output_dir / "algorithm_counts.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(records)} nuclei)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
