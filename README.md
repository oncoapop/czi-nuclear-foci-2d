# CZI Nuclear Foci Analysis (2-D)

Config-driven local analysis for 3-channel ZEISS CZI images:

- nuclear segmentation from one channel;
- independent segmentation of two nuclear DNA-damage focus channels;
- co-localisation calls between the two focus channels inside nuclei;
- mask TIFFs and channel-specific QC PNGs for manual review;
- per-image, per-nucleus, per-focus and co-localised-pair CSV outputs.

Raw CZI files are never modified and no image data is uploaded.

**This repository is the 2-D analysis.** The 3-D / Imaris (`.ims`) work lives in
[oncoapop/czi-nuclear-foci](https://github.com/oncoapop/czi-nuclear-foci). The two were split
because 3-D segmentation genuinely differs (ball footprints, volume thresholds, anisotropic
voxels, physical-distance colocalisation), not for tidiness. Both carry their own copy of
`czi_reader.py`; fixes to the binary CZI parsing need porting between them.

Tag **`v1.0-2d-published`** marks the exact code and configuration that produced the published
TS III / TS IV / TS V results. A fresh clone at that tag reproduces all 225 published fields
bit-exactly. Start there if you are checking published numbers.

## Environment - read this before installing

**Do not create a fresh venv and `pip install -e .`.** Newer scikit-image changes
`peak_local_max`, `watershed` and `threshold_otsu` behaviour, and segmentation output changes
with it. Results were produced with, and only reproduce under:

| package | version |
|---|---|
| python | 3.11.4 |
| numpy | 1.26.2 |
| scipy | 1.13.1 |
| scikit-image | **0.24.0** |
| pandas | 2.1.4 |
| matplotlib | 3.8.2 |
| reportlab | 4.0.8 |

Every command below assumes:

```zsh
env PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib <your-pinned-python> ...
```

`PYTHONPATH=src` is required. An editable install writes an absolute path into its `.pth`, so
without it a copy of this project silently imports the *original* checkout's code.

## Run everything with one command

```zsh
env PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib python3 -m czi_foci.cli run \
  --config configs/time_series.yaml
```

One YAML drives segmentation of every experiment, the cross-experiment combine, the batch-aware
report, and a `run_manifest.json` recording the config, git commit, interpreter and package
versions. To analyse a new dataset, copy `configs/time_series.yaml`, change `run.output_root`
and the `experiments` block, and check `filename.patterns` matches your filenames.

The combine step matters: the combined cross-experiment table that every downstream result
depends on previously had no generating script, so the plate-code mapping and control assignment
were untracked. Both now come from the config.

YAML needs PyYAML; the same config works as JSON with no extra dependency.

## How to report results

The 2026-09-15 audit measured what survives a change to the detection threshold
(`threshold_mad_multiplier`, the dominant parameter):

- **Report fold-changes against within-batch controls.** Ratios are stable across multiplier
  3-5 (median spread 1.23x, direction stable in 30/35 condition-blocks).
- **Do not report absolute counts.** Moving the multiplier 3 to 5 changes total 53BP1 counts
  1.32x to 0.75x. "15 foci per nucleus" is as much a statement about the multiplier as about
  the biology, and is not comparable to another lab's absolute counts.
- **Do not claim "drug X > drug Y".** Between-drug ranking is not robust even where
  treated-vs-control is. In TS IV 24 h the order is Etoposide > CX-5461 > PDS at multiplier
  3.0 but CX-5461 > PDS > Etoposide at 5.0.
- Four fields per condition is underpowered for FDR-corrected significance. Large, consistent
  effects can still read "ns".

## Channel status

Benchmarked against literature expectation, using Palbociclib (CDK4/6 inhibitor, non-genotoxic)
as a negative anchor and Etoposide (TOP2 poison) as a positive one:

| channel | verdict |
|---|---|
| **53BP1 / AF488** | **Usable.** Palbociclib matches vehicle in 7/7 blocks; all genotoxins elevated; Cisplatin shows its known slow kinetics. |
| gamma-H2AX / RhReX | **Failed QC.** Palbociclib significantly elevated in 3/7 blocks; Etoposide ranks last in TS III 6 h. No drug response by foci count *or* nuclear intensity - a staining/specificity problem, not an analysis one. |
| Colocalised | **Do not report.** Inherits the gamma-H2AX failure, and is additionally a geometric artefact: 39-68% of RhReX foci are called colocalised, tracking AF488 density. Needs a randomisation control. |

## Config reference

- `nuclei.channel_index`: nuclear marker, usually DAPI.
- `focus_a.channel_index` / `focus_b.channel_index`: the two focus channels.
- `focus_a.name` / `focus_b.name`: short output-safe names, e.g. `AF488`, `gH2AX`, `53BP1`.
- `threshold_mad_multiplier`: how many robust SDs above the typical nuclear pixel a pixel must
  be to count as focus signal. **The dominant parameter** - see "How to report results".
- `min_intensity_above_local_background`: **verified inert.** Values 4.0, 8.0 and 12.0 give
  byte-identical output; even 40.0 shifts totals by 0.4%. Retained at the published value for
  reproducibility, not because it contributes.
- `colocalization_dilation_px`: pixel tolerance for overlap.
- `conditions.controls` / `conditions.map`: control list, and plate-code to treatment mapping.
- `analysis.batch_columns`: defaults to `[experiment, timepoint_hr]`. **Not `[experiment]`** -
  TS III control levels fall 1.63x between 2 h and 6 h (Welch p = 0.0004) on a single
  acquisition date, so experiment-only batching under-corrects it.

## Output files

Per run: `resolved_config.json`, `image_summary.csv`, `nucleus_measurements.csv`,
`<focus>_measurements.csv`, `colocalized_focus_pairs.csv`, `condition_summary.csv`,
`threshold_performance_<metric>.csv`, `best_thresholds.csv`, `masks/*.tif`,
`qc_overlays/*.png`, `qc_channels/*.png`, `qc_contact_sheet.pdf`.

Combined: `combined_{nucleus_measurements,image_summary}.csv`, the `batch_aware/` tables
(`control_stats`, `normalised_nuclei`, `raw_/normalised_best_thresholds_by_batch`,
`*_threshold_stability`, `effect_sizes`, `proportion_positive_by_condition`), and
`run_manifest.json`.

`effect_sizes.csv` is the one to use for biological interpretation when cut-offs are not stable
across batches.

## Limitations

- **Multi-plane CZI is refused by design.** This reader keeps one plane per channel. Before the
  guard was added it silently kept the *last* Z-plane - a 19-plane stack collapsed to its dim
  terminal plane with no error. Project to a single plane, or use the 3-D repository.
- **TRF2Opt (16 files) is not covered.** Two channels only (TRF2 + DAPI), no second focus
  channel. Analysed separately by `segment_trf2_dapi.py`. Running it here needs optional-`focus_b`
  support in the runner.
- The parser supports uncompressed Gray8/Gray16 only - no JPEG XR, mosaics or pyramids.
- Threshold performance treats all non-control conditions as positive examples. That is
  screening-level separation from background, not evidence of mechanism.
- Cross-experiment comparisons remain vulnerable to staining and acquisition effects unless
  normalised within batch.

## Provenance

`docs/PROVENANCE.md` records what produced what, the required interpreter, data coverage
(225 of 250 CZI files), and the per-channel verdicts. Note that the published results came from
`segment_dual_channel_coloc.py`, a root script that used to be gitignored - which is why their
provenance could not previously be reconstructed from a clone. All root scripts are now tracked.

## Verify

```zsh
env PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib python3 -m unittest discover -s tests   # 24 tests
env PYTHONPATH=src python3 audit_diff_repro.py --rerun <new-output> --published <published-output>
```

`audit_diff_repro.py` compares a re-run against published output field by field and nucleus by
nucleus, exactly. Use it after any environment or dependency change.
