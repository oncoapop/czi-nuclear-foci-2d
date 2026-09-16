# Provenance, environment and known limits

Written after the 2026-09-15 audit. Full findings: `AUDIT_REPORT.md` in the audit output directory.

## What produced the published TS III/IV/V results

The root script **`segment_dual_channel_coloc.py`**, not the packaged `czi-foci` CLI. Confirmed
because each published output directory contains a `parameters.json` matching that script's
`DualParameters` dataclass, which the package never writes.

That script, and `segment_nuclei_spots.py` which it loads, were previously gitignored. They are now
tracked. Keeping them out of version control is the reason the published results could not be
reproduced from a clone.

The packaged code is numerically identical to the legacy script - verified component by component
(channel reads, nuclei, both spot channels including thresholds, colocalisation) - so
`czi-foci run` is a safe replacement going forward.

## Environment

Reproduction requires:

```zsh
env PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib /Users/dyap/miniconda3/bin/python3 ...
```

numpy 1.26.2, scipy 1.13.1, scikit-image 0.24.0, pandas 2.1.4, matplotlib 3.8.2, reportlab 4.0.8.

The in-project `.venv` has numpy 2.x / scikit-image 0.26 and **no matplotlib**. It cannot have
produced the published results and must not be used - `peak_local_max`, `watershed` and
`threshold_otsu` semantics changed across those versions.

`PYTHONPATH=src` is required: the editable-install `.pth` carries an absolute path, so without it a
copy of this project silently imports the original's code.

## Data coverage

250 CZI files exist under `/Volumes/Backup/TS Runs`. 225 are covered by the 2-D pipeline:

| dataset | files | status |
|---|---|---|
| Time Series III | 64 | analysed |
| Time Series IV | 97 | analysed (1 field has zero nuclei and is excluded by rule) |
| Time Series V | 64 | analysed; wells A-H mapped to treatments via config |
| **TRF2Opt** | 16 | **not in this pipeline** - 2 channels only (TRF2 + DAPI), no second focus channel. Analysed separately by `segment_trf2_dapi.py`; see `output/trf2opt_trf2_dapi*`. Running it here needs optional-`focus_b` support. |
| **TRF2 + yH2AX coloc** | 9 | **not 2-D** - 3 channels x 19 Z-planes, 16-bit. Belongs to the 3-D analysis mode. |

## Known limits

- **Multi-plane CZI is refused, by design.** `io.read_channel_arrays` keeps one plane per channel.
  Before the guard was added it silently kept the *last* Z-plane: the TRF2+yH2AX files collapsed
  57 subblocks to 3, discarding 54 planes and retaining a dim terminal plane (channel 0 first-plane
  mean 1570 vs last-plane mean 516) with no error. Project to a single plane, or use 3-D mode.
- **`min_intensity_above_local_background` does nothing.** Values 4.0, 8.0 and 12.0 give
  byte-identical output; even 40.0 shifts totals by 0.4%. The MAD threshold is the only effective
  gate. Retained at 8.0 for reproducibility, not because it contributes.
- **Absolute foci counts are parameter-dependent.** Moving `threshold_mad_multiplier` from 3 to 5
  changes total 53BP1 counts 1.32x -> 0.75x. Ratios to within-batch controls are stable
  (median spread 1.23x, direction stable in 30/35 condition-blocks). **Report fold-changes.**
- **Between-drug ranking is not robust** even where treated-vs-control is. Report "drug X vs
  control", not "drug X > drug Y".
- **Four fields per condition is underpowered** for FDR-corrected significance. Large, consistent
  effects can still read "ns".

## Channel trust, as of this audit

Benchmarked against literature expectation, using Palbociclib (CDK4/6 inhibitor, non-genotoxic) as
a negative anchor and Etoposide (TOP2 poison) as a positive one.

| channel | verdict |
|---|---|
| **53BP1 / AF488** | **Usable.** Palbociclib matches vehicle in 7/7 blocks; all genotoxins elevated; Cisplatin shows its known slow kinetics. |
| gamma-H2AX / RhReX | **Failed QC.** Palbociclib significantly elevated in 3/7 blocks; Etoposide ranks last in TS III 6 h. No drug response by foci count *or* nuclear intensity. A staining/specificity problem, not an analysis one. |
| Colocalised | **Do not report.** Inherits the gamma-H2AX failure, and is additionally a geometric artefact - 39-68% of RhReX foci are called colocalised, tracking AF488 density. Needs a randomisation control (rotate/shift one mask, report observed minus expected). |

## Batching

`batch_columns` defaults to `[experiment, timepoint_hr]`, not `[experiment]`. TS III control levels
fall 1.63x between 2 h and 6 h (Welch p = 0.0004) on a single acquisition date - nearly the whole
between-experiment range (1.95x). Normalising per experiment alone pools those baselines and
under-corrects TS III. The published `batch_aware_current_dataset/` tables use the coarser key.
