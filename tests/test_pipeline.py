from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from czi_foci.pipeline import metric_names, proportion_summary, resolve_pattern
from czi_foci.run_config import ExperimentSpec
from czi_foci.config import AnalysisConfig, FilenameConfig, FocusConfig, NucleiConfig

PATTERNS = [
    r"^(?P<timepoint>\d+)\s*hr\s+20x\s+(?P<condition>.+)-(?P<replicate>\d+)\.czi$",   # TS IV / TS V
    r"^(?P<timepoint>\d+)\s*hr\s+(?P<condition>.+?)\s+20x\s*-?\s*(?P<replicate>\d+)\.czi$",  # TS III
]


def spec(a="af488_spot", b="rhrex_putative_gH2AX") -> ExperimentSpec:
    cfg = AnalysisConfig(
        experiment_name="E", nuclei=NucleiConfig(),
        focus_a=FocusConfig(name=a, channel_index=1, label=a, tophat_radius_px=4, min_area_px=3),
        focus_b=FocusConfig(name=b, channel_index=0, label=b, tophat_radius_px=8, min_area_px=8),
        filename=FilenameConfig(regex=PATTERNS[0]))
    return ExperimentSpec(name="E", root=Path("/tmp"), analysis=cfg, patterns=PATTERNS)


class MetricNameTests(unittest.TestCase):
    def test_metrics_follow_the_focus_names(self) -> None:
        self.assertEqual(
            metric_names(spec()),
            ["af488_spot_count", "rhrex_putative_gH2AX_count",
             "colocalized_rhrex_putative_gH2AX_count"])

    def test_renaming_a_channel_renames_its_metrics(self) -> None:
        self.assertEqual(metric_names(spec(a="53BP1", b="gH2AX")),
                         ["53BP1_count", "gH2AX_count", "colocalized_gH2AX_count"])


class ResolvePatternTests(unittest.TestCase):
    def test_picks_the_pattern_matching_most_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["2 hr Aqueous 20x -01.czi", "2 hr Aqueous 20x -02.czi",
                         "6 hr DMSO 20x -01.czi"]:                      # TS III convention
                (root / name).touch()
            (root / "2 hr 20x Aqueous-01.czi").touch()                  # single TS IV-style file
            chosen, files = resolve_pattern(root, PATTERNS)
            self.assertEqual(chosen, PATTERNS[1])
            self.assertEqual(len(files), 3)

    def test_no_czi_files_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                resolve_pattern(Path(tmp), PATTERNS)

    def test_files_matching_nothing_are_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "unexpected_name.czi").touch()
            with self.assertRaises(SystemExit):
                resolve_pattern(Path(tmp), PATTERNS)


class ProportionSummaryTests(unittest.TestCase):
    def test_fraction_positive_is_computed_within_batch(self) -> None:
        nuclei = pd.DataFrame({
            "experiment": ["E"] * 8,
            "timepoint_hr": [2] * 4 + [6] * 4,
            "condition": ["DMSO", "DMSO", "Eto", "Eto"] * 2,
            "known_control": [True, True, False, False] * 2,
            "m": [0, 0, 1, 3, 0, 2, 4, 5],
        })
        out = proportion_summary(nuclei, ["m"], ["experiment", "timepoint_hr"])
        row = out[(out.timepoint_hr == 2) & (out.condition == "Eto")].iloc[0]
        self.assertEqual(row["fraction_ge1_m"], 1.0)
        self.assertEqual(row["mean_m"], 2.0)
        ctl = out[(out.timepoint_hr == 2) & (out.condition == "DMSO")].iloc[0]
        self.assertEqual(ctl["fraction_ge1_m"], 0.0)
        self.assertEqual(len(out), 4)          # 2 timepoints x 2 conditions, batches kept separate


if __name__ == "__main__":
    unittest.main()
