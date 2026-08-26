from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "summarize_boundary_factor_trajectory.py"
SPEC = importlib.util.spec_from_file_location("boundary_factor_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_quantile_uses_linear_interpolation() -> None:
    assert MODULE.quantile([0.0, 1.0], 0.2) == 0.2
    assert MODULE.quantile([0.0, 1.0], 0.8) == 0.8


def test_step100_windows_are_not_named_terminal() -> None:
    assert MODULE.window_name(10, 100) == "early"
    assert MODULE.window_name(40, 100) == "middle"
    assert MODULE.window_name(70, 100) == "late"


def test_constraint_directions_are_non_compensating() -> None:
    control = [{
        "label": "control",
        "step": 60,
        "category_fracture_record_occupancy": 0.10,
        "family_coverage": 0.60,
        "landmark_coverage": 0.50,
        "family_integrity": 0.80,
    }]
    variant = [{
        "label": "variant",
        "step": 60,
        "category_fracture_record_occupancy": 0.16,
        "family_coverage": 0.56,
        "landmark_coverage": 0.44,
        "family_integrity": 0.76,
    }]
    rows = MODULE.constraint_rows(variant, control)
    violations = {row["metric"]: row["violated"] for row in rows}
    assert violations == {
        "category_fracture_record_occupancy": True,
        "family_coverage": False,
        "landmark_coverage": True,
        "family_integrity": False,
    }
