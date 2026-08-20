import json
from pathlib import Path

import pytest
import torch

from hnet.training.linguistic_boundaries import (
    FocusAnnotation,
    acceptable_byte_offsets,
    boundary_budget,
    occurrence_byte_span,
    protected_byte_ranges,
    score_focus_boundaries,
    segmentation_byte_offsets,
    select_topk_boundary_mask,
)


def test_segmentation_offsets_use_utf8_bytes() -> None:
    assert segmentation_byte_offsets("笑って|いる", "笑っている") == {9}
    assert segmentation_byte_offsets("笑|っている", "笑っている") == {3}


def test_segmentation_must_reconstruct_surface() -> None:
    with pytest.raises(ValueError, match="does not reconstruct"):
        segmentation_byte_offsets("笑|いる", "笑っている")


def test_occurrence_byte_span_handles_repeated_japanese_surface() -> None:
    text = "猫が笑う。別の猫が眠る。"
    assert occurrence_byte_span(text, "猫", 1) == (
        len("猫が笑う。別の".encode("utf-8")),
        len("猫が笑う。別の猫".encode("utf-8")),
    )


def test_topk_boundary_mask_preserves_required_start() -> None:
    probability = torch.tensor([0.01, 0.2, 0.9, 0.8, 0.7])
    valid = torch.tensor([True, True, True, True, False])
    required = torch.tensor([True, False, False, False, False])
    selected = select_topk_boundary_mask(probability, valid, 3, required)
    assert selected.tolist() == [True, False, True, True, False]


def test_boundary_budget_uses_ceiling_and_preserves_start() -> None:
    assert boundary_budget(10, 3.0) == 4
    assert boundary_budget(1, 3.0) == 1


def test_focus_score_excludes_mid_codepoint_boundaries() -> None:
    text = "彼女は笑っている。"
    annotation = FocusAnnotation(
        surface="笑っている",
        occurrence=0,
        acceptable_segmentations=("笑|っている", "笑って|いる"),
    )
    focus_start = len("彼女は".encode("utf-8")) + 1
    positions = [
        focus_start + 1,
        focus_start + 3,
        focus_start + 6,
    ]
    score = score_focus_boundaries(text, annotation, positions)
    assert score["constraint_dependent_offsets"] == [1]
    assert score["explained_offsets"] == [3]
    assert score["unexplained_offsets"] == [6]
    assert score["explainable_boundary_precision"] == 0.5
    assert score["unexplained_boundary_rate"] == 0.5


def test_short_fragment_run_is_reported() -> None:
    text = "結果は人が確認した。"
    annotation = FocusAnnotation(
        surface="結果は人が",
        occurrence=0,
        acceptable_segmentations=("結果|は|人|が",),
    )
    offsets = [
        len(prefix.encode("utf-8")) + 1
        for prefix in ("結", "結果", "結果は", "結果は人")
    ]
    score = score_focus_boundaries(text, annotation, offsets)
    assert score["has_pathological_fragmentation"] is True
    assert score["pathological_fragmentation_runs"] == 1
    assert score["has_short_fragmentation"] is True
    assert score["has_severe_short_fragmentation"] is False


def test_four_unexplained_short_fragments_are_severe() -> None:
    text = "自然言語処理を学ぶ。"
    annotation = FocusAnnotation(
        surface="自然言語処理",
        occurrence=0,
        acceptable_segmentations=("自然|言語|処理",),
        protected_substrings=("自然", "言語"),
    )
    offsets = [
        len(prefix.encode("utf-8")) + 1
        for prefix in ("自", "自然", "自然言", "自然言語")
    ]
    score = score_focus_boundaries(text, annotation, offsets)
    assert score["has_short_fragmentation"] is True
    assert score["has_severe_short_fragmentation"] is True
    assert score["lexeme_fracture_offsets"] == [3, 9]


def test_protected_ranges_and_best_complete_segmentation() -> None:
    annotation = FocusAnnotation(
        surface="東京都千代田区",
        occurrence=0,
        acceptable_segmentations=("東京都|千代田区", "東京|都|千代田|区"),
        protected_substrings=("東京", "千代田"),
    )
    assert protected_byte_ranges(annotation) == [(0, 6), (9, 18)]
    positions = [len(prefix.encode("utf-8")) + 1 for prefix in ("東", "東京都")]
    score = score_focus_boundaries(
        "東京都千代田区で会議を開く。", annotation, positions
    )
    assert score["best_segmentation"]["segmentation"] == "東京都|千代田区"
    assert score["best_segmentation"]["f1"] == pytest.approx(2.0 / 3.0)


def test_probe_segmentations_are_well_formed() -> None:
    root = Path(__file__).resolve().parents[2]
    probe = json.loads(
        (root / "configs/linguistic_boundary_probe_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(probe["records"]) >= 20
    for record in probe["records"]:
        focus = record["focus"]
        annotation = FocusAnnotation(
            surface=focus["surface"],
            occurrence=focus.get("occurrence", 0),
            acceptable_segmentations=tuple(focus["acceptable_segmentations"]),
        )
        occurrence_byte_span(record["text"], annotation.surface, annotation.occurrence)
        acceptable_byte_offsets(annotation)


def test_v2_probe_segmentations_and_protected_substrings_are_well_formed() -> None:
    root = Path(__file__).resolve().parents[2]
    probe = json.loads(
        (root / "configs/linguistic_boundary_probe_v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert probe["version"] == 2
    assert len(probe["records"]) == 88
    counts: dict[str, int] = {}
    for record in probe["records"]:
        counts[record["category"]] = counts.get(record["category"], 0) + 1
        focus = record["focus"]
        annotation = FocusAnnotation(
            surface=focus["surface"],
            occurrence=focus.get("occurrence", 0),
            acceptable_segmentations=tuple(focus["acceptable_segmentations"]),
            protected_substrings=tuple(focus.get("protected_substrings", ())),
        )
        occurrence_byte_span(record["text"], annotation.surface, annotation.occurrence)
        acceptable_byte_offsets(annotation)
        protected_byte_ranges(annotation)
    assert set(counts.values()) == {8}
