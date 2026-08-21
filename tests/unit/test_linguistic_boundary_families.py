import json
from pathlib import Path

import pytest

from hnet.training.linguistic_boundary_families import (
    FamilyLandmark,
    landmark_byte_offset,
    summarize_family_scores,
    validate_family_annotation,
)
from hnet.training.linguistic_boundary_trajectory import (
    checkpoint_order,
    summarize_trajectory_scores,
)


def _score(
    selected: list[int], explained: list[int], *, fractured: bool = False
) -> dict:
    return {
        "selected_evaluable_offsets": selected,
        "explained_offsets": explained,
        "unexplained_offsets": sorted(set(selected) - set(explained)),
        "acceptable_offsets": [6],
        "lexeme_fracture_offsets": [3] if fractured else [],
        "best_segmentation": {"f1": 1.0 if selected == [6] else 0.0},
    }


def _record(record_id: str, selected: list[int], *, text: str = "文を分割する。") -> dict:
    score = _score(selected, [offset for offset in selected if offset == 6])
    return {
        "id": record_id,
        "category": "inflection",
        "text": text,
        "focus": {
            "surface": "分割する",
            "acceptable_segmentations": ["分割|する"],
        },
        "family": {
            "group": "split-control",
            "kind": "context_control",
            "landmarks": [
                {"name": "lexeme_suffix", "segmentation": "分割|する"}
            ],
        },
        "conditions": {
            "central": {
                "stage0": {"score": score},
                "stage1": {"score": score},
            }
        },
    }


def _run(step: int, records: list[dict]) -> dict:
    return {
        "version": 3,
        "model_name": "K3G1",
        "seed": 42,
        "checkpoint_label": f"step_{step:06d}",
        "records": records,
    }


def test_landmark_offset_uses_utf8_bytes() -> None:
    assert landmark_byte_offset(
        "分割する", FamilyLandmark("lexeme_suffix", "分割|する")
    ) == 6


def test_landmark_requires_exactly_one_boundary() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        landmark_byte_offset("分割する", FamilyLandmark("bad", "分|割|する"))


def test_family_consistency_does_not_hide_zero_coverage() -> None:
    run = _run(55, [_record("a", []), _record("b", [])])
    family_rows, landmark_rows = summarize_family_scores([run])
    assert family_rows[0]["category_coverage"] == 0.0
    assert landmark_rows[0]["landmark_coverage"] == 0.0
    assert landmark_rows[0]["landmark_consistency"] == 1.0
    assert landmark_rows[0]["none_selected"] is True


def test_context_family_reports_signature_consistency() -> None:
    run = _run(55, [_record("a", [6]), _record("b", [6]), _record("c", [])])
    family_rows, landmark_rows = summarize_family_scores([run])
    assert family_rows[0]["context_signature_consistency"] == pytest.approx(2 / 3)
    assert landmark_rows[0]["landmark_coverage"] == pytest.approx(2 / 3)
    assert landmark_rows[0]["landmark_consistency"] == pytest.approx(2 / 3)


def test_family_probe_annotations_are_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    probe = json.loads(
        (root / "configs/linguistic_boundary_family_probe_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert probe["version"] == 3
    assert len(probe["records"]) == 24
    groups: dict[str, int] = {}
    for record in probe["records"]:
        validate_family_annotation(record)
        groups[record["family"]["group"]] = groups.get(
            record["family"]["group"], 0
        ) + 1
    assert len(groups) == 6
    assert set(groups.values()) == {4}


def test_sparse_trajectory_reports_transitions_and_late_window() -> None:
    runs = [
        _run(55, [_record("a", [])]),
        _run(110, [_record("a", [6])]),
        _run(165, [_record("a", [6])]),
    ]
    rows = summarize_trajectory_scores(runs)
    overall = next(
        row
        for row in rows
        if row["category"] == "__all__" and row["stage"] == "stage1"
    )
    assert overall["checkpoint_count"] == 3
    assert overall["segmentation_transition_rate"] == pytest.approx(0.5)
    assert overall["late_coverage"] == 1.0
    assert overall["late_checkpoint_count"] == 2


def test_checkpoint_order_uses_last_numeric_component() -> None:
    assert checkpoint_order("seed42_step_000220") == 220
