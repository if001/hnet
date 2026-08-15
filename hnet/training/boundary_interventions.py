from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

import torch

from hnet.modules.dc import BoundaryOverride, RoutingModuleOutput


def utf8_safe_positions(token_ids: Sequence[int]) -> list[int]:
    """Return token positions that do not start on a UTF-8 continuation byte."""

    return [
        index
        for index, token_id in enumerate(token_ids)
        if not 0x80 <= int(token_id) <= 0xBF
    ]


def evenly_spaced_positions(candidates: Sequence[int], count: int) -> list[int]:
    """Select exactly ``count`` positions distributed across candidates."""

    ordered = sorted(set(int(value) for value in candidates))
    if count < 1 or count > len(ordered):
        raise ValueError(
            f"Cannot select {count} boundaries from {len(ordered)} candidates"
        )
    if count == 1:
        return [ordered[0]]

    indices = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    return [ordered[index] for index in indices]


def random_positions(
    candidates: Sequence[int],
    count: int,
    seed: int,
) -> list[int]:
    """Sample positions while always retaining the first candidate."""

    ordered = sorted(set(int(value) for value in candidates))
    if count < 1 or count > len(ordered):
        raise ValueError(
            f"Cannot select {count} boundaries from {len(ordered)} candidates"
        )
    if count == 1:
        return [ordered[0]]
    rng = random.Random(seed)
    return [ordered[0], *sorted(rng.sample(ordered[1:], count - 1))]


def prioritized_positions(
    candidates: Sequence[int],
    priorities: Iterable[int],
    count: int,
) -> list[int]:
    """Prefer supplied boundaries, then fill remaining slots evenly."""

    ordered = sorted(set(int(value) for value in candidates))
    candidate_set = set(ordered)
    preferred = sorted(
        {int(value) for value in priorities if int(value) in candidate_set}
        | {ordered[0]}
    )
    if count < 1 or count > len(ordered):
        raise ValueError(
            f"Cannot select {count} boundaries from {len(ordered)} candidates"
        )
    if len(preferred) > count:
        return evenly_spaced_positions(preferred, count)
    if len(preferred) == count:
        return preferred

    remaining = [value for value in ordered if value not in set(preferred)]
    fill = evenly_spaced_positions(remaining, count - len(preferred))
    return sorted([*preferred, *fill])


def shifted_positions(
    learned: Sequence[int],
    candidates: Sequence[int],
    direction: int,
) -> list[int]:
    """Move learned boundaries by one candidate, preserving count and start."""

    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    ordered = sorted(set(int(value) for value in candidates))
    candidate_indices = {value: index for index, value in enumerate(ordered)}
    learned_ordered = sorted(set(int(value) for value in learned))
    if not learned_ordered or learned_ordered[0] != ordered[0]:
        raise ValueError("Learned boundaries must contain the first candidate")
    if any(value not in candidate_indices for value in learned_ordered):
        raise ValueError("Learned boundary is absent from candidate positions")

    selected = {ordered[0]}
    desired: list[int] = []
    for value in learned_ordered[1:]:
        index = candidate_indices[value]
        shifted_index = min(max(index + direction, 1), len(ordered) - 1)
        desired.append(ordered[shifted_index])

    for value in desired:
        available = [candidate for candidate in ordered[1:] if candidate not in selected]
        selected.add(min(available, key=lambda candidate: (abs(candidate - value), candidate)))
    return sorted(selected)


def morphological_byte_positions(text: str, add_bos: bool = True) -> list[int]:
    """Return fugashi word starts/ends expressed as model token positions."""

    try:
        from fugashi import Tagger
    except ImportError as exc:  # pragma: no cover - depends on Colab preparation
        raise RuntimeError(
            "Morph evaluation requires fugashi and a MeCab dictionary; install "
            "fugashi[unidic-lite]."
        ) from exc

    positions = {0}
    cursor = 0
    raw = text.encode("utf-8")
    bos_offset = 1 if add_bos else 0
    for word in Tagger()(text):
        surface = str(word.surface)
        if not surface:
            continue
        surface_bytes = surface.encode("utf-8")
        start = raw.find(surface_bytes, cursor)
        if start < 0:
            continue
        end = start + len(surface_bytes)
        positions.add(start + bos_offset)
        if end < len(raw):
            positions.add(end + bos_offset)
        cursor = end
    return sorted(positions)


def mask_from_positions(
    positions: Sequence[int],
    length: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((1, length), dtype=torch.bool, device=device)
    mask[0, list(positions)] = True
    return mask


def transplant_boundary_probabilities(
    learned: RoutingModuleOutput,
    proposed_mask: torch.Tensor,
) -> torch.Tensor:
    """Move learned selected-boundary confidences to proposed positions in order."""

    proposed_mask = proposed_mask.to(
        device=learned.boundary_mask.device, dtype=torch.bool
    )
    if proposed_mask.shape != learned.boundary_mask.shape:
        raise ValueError("Proposed boundary mask shape does not match learned mask")

    learned_mask = learned.boundary_mask
    learned_probability = learned.boundary_prob
    probabilities = learned.boundary_prob.clone()
    if learned_mask.ndim == 1:
        learned_mask = learned_mask.unsqueeze(0)
        learned_probability = learned_probability.unsqueeze(0)
        proposed_mask = proposed_mask.unsqueeze(0)
        probabilities = probabilities.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    for batch_index in range(learned_mask.shape[0]):
        learned_values = learned_probability[
            batch_index, learned_mask[batch_index], 1
        ]
        proposed_count = int(proposed_mask[batch_index].sum().item())
        if proposed_count != learned_values.numel():
            raise ValueError(
                "Counterfactual boundary count must match learned boundary count"
            )
        probabilities[batch_index, proposed_mask[batch_index], 1] = learned_values
        probabilities[batch_index, proposed_mask[batch_index], 0] = 1.0 - learned_values

    return probabilities.squeeze(0) if squeeze else probabilities


def make_boundary_override(
    learned: RoutingModuleOutput,
    proposed_mask: torch.Tensor,
) -> BoundaryOverride:
    return BoundaryOverride(
        boundary_mask=proposed_mask,
        boundary_prob=transplant_boundary_probabilities(learned, proposed_mask),
    )
