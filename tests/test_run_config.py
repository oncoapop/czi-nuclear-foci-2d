from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from czi_foci.run_config import load_run_config

BASE = {
    "version": 1,
    "run": {"output_root": "out", "combine": {"enabled": True, "batch_aware_report": False}},
    "defaults": {
        "channels": {
            "nuclei": {"index": 2, "label": "DAPI"},
            "focus_a": {"name": "af488_spot", "index": 1, "tophat_radius_px": 4, "min_area_px": 3},
            "focus_b": {"name": "rhrex", "index": 0, "tophat_radius_px": 8, "min_area_px": 8},
        },
        "filename": {"patterns": [r"^(?P<timepoint>\d+)hr_(?P<condition>.+)_(?P<replicate>\d+)\.czi$"]},
        "conditions": {"controls": ["DMSO"]},
        "analysis": {"batch_columns": ["experiment", "timepoint_hr"]},
    },
    "experiments": [{"name": "E1", "root": "/tmp/e1"}],
}


def write(data: dict, suffix: str = ".json") -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
    if suffix == ".json":
        json.dump(data, handle)
    else:
        import yaml
        yaml.safe_dump(data, handle)
    handle.close()
    return Path(handle.name)


class RunConfigTests(unittest.TestCase):
    def test_defaults_are_inherited_by_experiments(self) -> None:
        spec = load_run_config(write(BASE))
        exp = spec.experiments[0]
        self.assertEqual(exp.name, "E1")
        self.assertEqual(exp.analysis.focus_a.name, "af488_spot")
        self.assertEqual(exp.analysis.focus_a.threshold_mad_multiplier, 4.0)  # documented default
        self.assertEqual(spec.batch_columns, ["experiment", "timepoint_hr"])

    def test_experiment_overrides_beat_defaults(self) -> None:
        data = json.loads(json.dumps(BASE))
        data["experiments"].append({
            "name": "E2", "root": "/tmp/e2",
            "channels": {"focus_a": {"name": "af488_spot", "index": 1,
                                     "tophat_radius_px": 9, "min_area_px": 3}},
        })
        spec = load_run_config(write(data))
        by_name = {e.name: e for e in spec.experiments}
        self.assertEqual(by_name["E1"].analysis.focus_a.tophat_radius_px, 4)
        self.assertEqual(by_name["E2"].analysis.focus_a.tophat_radius_px, 9)

    def test_condition_map_resolves_codes_and_control_flags(self) -> None:
        data = json.loads(json.dumps(BASE))
        data["experiments"][0]["conditions"] = {
            "controls": ["DMSO"],
            "map": {"A": {"condition": "Etoposide", "control": False},
                    "G": {"condition": "DMSO", "control": True}},
        }
        exp = load_run_config(write(data)).experiments[0]
        self.assertEqual(exp.resolve_condition("A"), ("Etoposide", False))
        self.assertEqual(exp.resolve_condition("G"), ("DMSO", True))
        # unmapped code falls through to the controls list
        self.assertEqual(exp.resolve_condition("DMSO"), ("DMSO", True))
        self.assertEqual(exp.resolve_condition("Novel"), ("Novel", False))

    def test_invalid_regex_fails_at_load_not_mid_run(self) -> None:
        data = json.loads(json.dumps(BASE))
        data["defaults"]["filename"]["patterns"] = ["^(?P<timepoint>\\d+"]
        with self.assertRaises(Exception):
            load_run_config(write(data))

    def test_missing_experiments_is_rejected(self) -> None:
        data = json.loads(json.dumps(BASE))
        data["experiments"] = []
        with self.assertRaises(ValueError):
            load_run_config(write(data))

    def test_yaml_and_json_agree(self) -> None:
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")
        a = load_run_config(write(BASE, ".json"))
        b = load_run_config(write(BASE, ".yaml"))
        self.assertEqual(a.batch_columns, b.batch_columns)
        self.assertEqual(a.experiments[0].analysis.focus_b.name,
                         b.experiments[0].analysis.focus_b.name)


if __name__ == "__main__":
    unittest.main()
