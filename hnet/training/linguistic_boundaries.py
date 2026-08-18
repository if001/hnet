from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class FocusAnnotation:
    surface: str
    occurrence: int
    acceptable_segmentations: tuple[str, ...]


def segmentation_byte_offsets(segmentation: str, surface: str) -> set[int]:
    """Return UTF-8 byte offsets denoted by ``|`` in a surface segmentation."""
    if segmentation.replace("|", "") != surface:
        raise ValueError(
            f"segmentation does not reconstruct surface: {segmentation!r} != {surface!r}"
        )
    offsets: set[int] = set()
    prefix = ""
    pieces = segmentation.split("|")
    if any(not piece for piece in pieces):
        raise ValueError("segmentation cannot contain an empty piece")
    for piece in pieces[:-1]:
        prefix += piece
        offsets.add(len(prefix.encode("utf-8")))
    return offsets


def occurrence_byte_span(text: str, surface: str, occurrence: int = 0) -> tuple[int, int]:
    """Locate a surface occurrence and return its UTF-8 byte span."""
    if occurrence < 0:
        raise ValueError("occurrence must be non-negative")
    start_character = 0
    for _ in range(occurrence + 1):
        found = text.find(surface, start_character)
        if found < 0:
            raise ValueError(
                f"surface occurrence {occurrence} is absent: {surface!r} in {text!r}"
            )
        start_character = found + 1
    character_start = found
    byte_start = len(text[:character_start].encode("utf-8"))
    return byte_start, byte_start + len(surface.encode("utf-8"))


def codepoint_boundary_offsets(surface: str) -> set[int]:
    offsets: set[int] = set()
    prefix = ""
    for character in surface[:-1]:
        prefix += character
        offsets.add(len(prefix.encode("utf-8")))
    return offsets


def acceptable_byte_offsets(annotation: FocusAnnotation) -> set[int]:
    offsets: set[int] = set()
    for segmentation in annotation.acceptable_segmentations:
        offsets.update(segmentation_byte_offsets(segmentation, annotation.surface))
    return offsets


def boundary_positions_from_mask(boundary_mask: Sequence[bool]) -> list[int]:
    return [index for index, selected in enumerate(boundary_mask) if selected]


def select_topk_boundary_mask(
    boundary_probability: torch.Tensor,
    valid_mask: torch.Tensor,
    boundary_count: int,
    required_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select an exact boundary budget using learned boundary probabilities."""
    if boundary_probability.shape != valid_mask.shape:
        raise ValueError("probability and valid mask shapes must match")
    if boundary_probability.ndim != 1:
        raise ValueError("top-k selection expects one sequence")
    if required_mask is None:
        required_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        required_mask[0] = True
    if required_mask.shape != valid_mask.shape:
        raise ValueError("required and valid mask shapes must match")
    required_mask = required_mask.to(dtype=torch.bool, device=valid_mask.device)
    valid_mask = valid_mask.to(dtype=torch.bool)
    if torch.any(required_mask & ~valid_mask):
        raise ValueError("required boundaries must be valid")

    valid_count = int(valid_mask.sum().item())
    required_count = int(required_mask.sum().item())
    if not required_count <= boundary_count <= valid_count:
        raise ValueError(
            f"boundary_count must be in [{required_count}, {valid_count}], got {boundary_count}"
        )

    selected = required_mask.clone()
    remaining = boundary_count - required_count
    if remaining == 0:
        return selected
    candidates = valid_mask & ~required_mask
    scores = boundary_probability.float().masked_fill(~candidates, -math.inf)
    top_indices = torch.topk(scores, k=remaining, sorted=False).indices
    selected[top_indices] = True
    return selected


def boundary_budget(sequence_length: int, units_per_chunk: float) -> int:
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if units_per_chunk <= 0:
        raise ValueError("units_per_chunk must be positive")
    return min(sequence_length, max(1, math.ceil(sequence_length / units_per_chunk)))


def _runs_of_short_fragments(
    surface: str,
    relative_boundaries: Iterable[int],
    maximum_codepoints: int = 1,
    minimum_run: int = 3,
) -> int:
    surface_bytes = surface.encode("utf-8")
    codepoint_offsets = codepoint_boundary_offsets(surface)
    usable = sorted(
        offset
        for offset in relative_boundaries
        if offset in codepoint_offsets
    )
    endpoints = [0, *usable, len(surface_bytes)]
    lengths: list[int] = []
    for start, end in zip(endpoints, endpoints[1:]):
        lengths.append(len(surface_bytes[start:end].decode("utf-8")))

    runs = 0
    current = 0
    for length in lengths:
        if length <= maximum_codepoints:
            current += 1
        else:
            if current >= minimum_run:
                runs += 1
            current = 0
    if current >= minimum_run:
        runs += 1
    return runs


def score_focus_boundaries(
    text: str,
    annotation: FocusAnnotation,
    boundary_positions: Sequence[int],
    *,
    input_prefix_tokens: int = 1,
) -> dict[str, object]:
    """Score model boundary starts inside one annotated surface span.

    Positions that split a UTF-8 codepoint are constraint-dependent and are reported
    separately. They are excluded from linguistic precision and unexplained rates.
    """
    byte_start, byte_end = occurrence_byte_span(
        text, annotation.surface, annotation.occurrence
    )
    model_start = byte_start + input_prefix_tokens
    model_end = byte_end + input_prefix_tokens
    relative = {
        position - model_start
        for position in boundary_positions
        if model_start < position < model_end
    }
    evaluable_offsets = codepoint_boundary_offsets(annotation.surface)
    acceptable_offsets = acceptable_byte_offsets(annotation)
    evaluable_selected = relative & evaluable_offsets
    constraint_dependent = relative - evaluable_offsets
    explained = evaluable_selected & acceptable_offsets
    unexplained = evaluable_selected - acceptable_offsets
    expected_selected = acceptable_offsets & evaluable_selected

    selected_count = len(evaluable_selected)
    acceptable_count = len(acceptable_offsets)
    precision = len(explained) / selected_count if selected_count else None
    coverage = len(expected_selected) / acceptable_count if acceptable_count else None
    unexplained_rate = len(unexplained) / selected_count if selected_count else None
    short_runs = _runs_of_short_fragments(annotation.surface, evaluable_selected)

    return {
        "surface": annotation.surface,
        "byte_span": [byte_start, byte_end],
        "acceptable_offsets": sorted(acceptable_offsets),
        "selected_evaluable_offsets": sorted(evaluable_selected),
        "explained_offsets": sorted(explained),
        "unexplained_offsets": sorted(unexplained),
        "constraint_dependent_offsets": sorted(constraint_dependent),
        "explainable_boundary_precision": precision,
        "category_coverage": coverage,
        "unexplained_boundary_rate": unexplained_rate,
        "pathological_fragmentation_runs": short_runs,
        "has_pathological_fragmentation": short_runs > 0,
    }
