"""Load a YAML (or JSON) pipeline config describing one or more experiments.

A single file drives the whole pipeline so that a new, similarly structured
dataset needs one command rather than a sequence of ad hoc scripts. Settings in
``defaults`` are inherited by every experiment and may be overridden per
experiment.

YAML support requires PyYAML; JSON works with no extra dependency.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import AnalysisConfig, FilenameConfig, FocusConfig, NucleiConfig, OutputConfig


@dataclass(frozen=True)
class ConditionInfo:
    condition: str
    control: bool = False
    concentration: str = ""


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    root: Path
    analysis: AnalysisConfig
    patterns: list[str]
    condition_map: dict[str, ConditionInfo] = field(default_factory=dict)
    controls: set[str] = field(default_factory=set)

    def resolve_condition(self, code: str) -> tuple[str, bool]:
        """Map a filename condition code to (name, is_control)."""
        info = self.condition_map.get(code)
        if info is not None:
            return info.condition, info.control
        return code, code in self.controls


@dataclass(frozen=True)
class RunSpec:
    output_root: Path
    experiments: list[ExperimentSpec]
    batch_columns: list[str]
    zero_inflated_metrics: list[str]
    exclude_border_nuclei: bool
    min_nuclei_per_field: int
    combine: bool
    batch_aware_report: bool
    write_qc_pdfs: bool
    source_path: Path
    raw: dict[str, Any]


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SystemExit(
                "Reading a YAML config needs PyYAML. Either install it:\n"
                "    python -m pip install --no-deps PyYAML\n"
                "or supply the same config as JSON, which needs no extra package."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _focus(block: dict, what: str) -> FocusConfig:
    missing = {"name", "index", "tophat_radius_px", "min_area_px"} - set(block)
    if missing:
        raise ValueError(f"channels.{what} is missing: {sorted(missing)}")
    return FocusConfig(
        name=str(block["name"]),
        channel_index=int(block["index"]),
        label=str(block.get("label", block["name"])),
        tophat_radius_px=int(block["tophat_radius_px"]),
        min_area_px=int(block["min_area_px"]),
        threshold_mad_multiplier=float(block.get("threshold_mad_multiplier", 4.0)),
        min_intensity_above_local_background=float(
            block.get("min_intensity_above_local_background", 8.0)
        ),
    )


def _experiment(name: str, merged: dict, config_dir: Path) -> ExperimentSpec:
    channels = merged.get("channels") or {}
    nuc = channels.get("nuclei") or {}
    nuclei = NucleiConfig(
        channel_index=int(nuc.get("index", 2)),
        label=str(nuc.get("label", "DAPI")),
        gaussian_sigma_px=float(nuc.get("gaussian_sigma_px", 1.2)),
        min_area_um2=float(nuc.get("min_area_um2", 35.0)),
        hole_area_um2=float(nuc.get("hole_area_um2", 20.0)),
        watershed_min_distance_px=int(nuc.get("watershed_min_distance_px", 10)),
    )
    focus_a = _focus(channels.get("focus_a") or {}, "focus_a")
    focus_b = _focus(channels.get("focus_b") or {}, "focus_b")

    filename = merged.get("filename") or {}
    patterns = list(filename.get("patterns") or ([filename["regex"]] if "regex" in filename else []))
    if not patterns:
        raise ValueError(f"experiment {name!r} has no filename.patterns")
    for pattern in patterns:
        re.compile(pattern)  # fail loudly at load time, not mid-run

    conditions = merged.get("conditions") or {}
    controls = {str(c) for c in (conditions.get("controls") or [])}
    condition_map: dict[str, ConditionInfo] = {}
    for code, spec in (conditions.get("map") or {}).items():
        if isinstance(spec, str):
            condition_map[str(code)] = ConditionInfo(condition=spec, control=spec in controls)
        else:
            condition_map[str(code)] = ConditionInfo(
                condition=str(spec["condition"]),
                control=bool(spec.get("control", str(spec["condition"]) in controls)),
                concentration=str(spec.get("concentration", "")),
            )

    qc = merged.get("qc") or {}
    output = OutputConfig(
        qc_display_percentile_low=float(qc.get("display_percentile_low", 1.0)),
        qc_display_percentile_high=float(qc.get("display_percentile_high", 99.8)),
    )

    analysis = AnalysisConfig(
        experiment_name=name,
        nuclei=nuclei,
        focus_a=focus_a,
        focus_b=focus_b,
        filename=FilenameConfig(regex=patterns[0]),
        control_conditions=sorted(controls),
        condition_order=[str(c) for c in (conditions.get("order") or [])],
        colocalization_dilation_px=int((merged.get("colocalization") or {}).get("dilation_px", 1)),
        output=output,
    )

    root = Path(str(merged["root"])).expanduser()
    if not root.is_absolute():
        root = (config_dir / root).resolve()
    return ExperimentSpec(
        name=name, root=root, analysis=analysis, patterns=patterns,
        condition_map=condition_map, controls=controls,
    )


def load_run_config(path: Path) -> RunSpec:
    path = path.resolve()
    data = _load_document(path)
    defaults = data.get("defaults") or {}
    entries = data.get("experiments") or []
    if not entries:
        raise ValueError(f"Config defines no experiments: {path}")

    experiments = []
    for entry in entries:
        if "name" not in entry or "root" not in entry:
            raise ValueError("Each experiment needs 'name' and 'root'")
        experiments.append(_experiment(str(entry["name"]), _deep_merge(defaults, entry), path.parent))

    run = data.get("run") or {}
    combine = run.get("combine") or {}
    analysis = defaults.get("analysis") or {}
    output_root = Path(str(run.get("output_root", "output/pipeline_run"))).expanduser()
    if not output_root.is_absolute():
        output_root = (path.parent / output_root).resolve()

    return RunSpec(
        output_root=output_root,
        experiments=experiments,
        batch_columns=[str(c) for c in (analysis.get("batch_columns") or ["experiment"])],
        zero_inflated_metrics=[str(m) for m in (analysis.get("zero_inflated_metrics") or [])],
        exclude_border_nuclei=bool(analysis.get("exclude_border_nuclei", False)),
        min_nuclei_per_field=int(analysis.get("min_nuclei_per_field", 1)),
        combine=bool(combine.get("enabled", True)),
        batch_aware_report=bool(combine.get("batch_aware_report", True)),
        write_qc_pdfs=bool(run.get("write_qc_pdfs", True)),
        source_path=path,
        raw=data,
    )
