import torch

from hnet.modules.dc import RoutingModuleOutput
from hnet.training.boundary_interventions import (
    evenly_spaced_positions,
    prioritized_positions,
    random_positions,
    shifted_positions,
    transplant_boundary_probabilities,
    utf8_safe_positions,
)


def test_utf8_safe_positions_excludes_continuation_bytes() -> None:
    assert utf8_safe_positions([254, 0xE7, 0x8C, 0xAB, ord("A")]) == [0, 1, 4]


def test_boundary_generators_preserve_count_and_start() -> None:
    candidates = [0, 1, 4, 7, 8, 9]
    learned = [0, 4, 8]

    generated = [
        evenly_spaced_positions(candidates, 3),
        random_positions(candidates, 3, seed=42),
        prioritized_positions(candidates, [0, 7], 3),
        shifted_positions(learned, candidates, direction=-1),
        shifted_positions(learned, candidates, direction=1),
    ]

    for positions in generated:
        assert len(positions) == 3
        assert positions[0] == 0
        assert set(positions) <= set(candidates)


def test_transplant_boundary_probabilities_preserves_confidence_order() -> None:
    learned_mask = torch.tensor([[True, False, True, False, True]])
    probability = torch.tensor(
        [[[0.0, 1.0], [0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]]
    )
    output = RoutingModuleOutput(
        boundary_prob=probability,
        boundary_mask=learned_mask,
        selected_probs=torch.ones((1, 5, 1)),
        valid_mask=torch.ones((1, 5), dtype=torch.bool),
    )
    proposed = torch.tensor([[True, True, False, True, False]])

    transplanted = transplant_boundary_probabilities(output, proposed)

    assert transplanted[0, proposed[0], 1].tolist() == [1.0, 0.8, 0.7]
