import pytest
import torch

from hnet.modules.dc import BoundaryOverride, RoutingModule, apply_boundary_override


def test_hard_constraint_disallows_continuation_byte_boundary() -> None:
    router = RoutingModule(d_model=2, device="cpu", dtype=torch.float32)
    hidden_states = torch.tensor(
        [[[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32
    )
    mask = torch.ones((1, 3), dtype=torch.bool)
    continuation_mask = torch.tensor([[False, True, False]])

    unconstrained = router(hidden_states, mask=mask)
    constrained = router(
        hidden_states,
        mask=mask,
        continuation_mask=continuation_mask,
        continuation_hard=True,
    )

    assert unconstrained.boundary_mask.tolist() == [[True, True, True]]
    assert constrained.boundary_mask.tolist() == [[True, False, True]]
    assert constrained.boundary_prob[0, 1].tolist() == [1.0, 0.0]
    assert constrained.selected_probs[0, 1].item() == 1.0


def test_recurrent_hard_constraint_disallows_continuation_byte_boundary() -> None:
    router = RoutingModule(d_model=2, device="cpu", dtype=torch.float32)
    state = router.allocate_inference_cache(
        batch_size=1, max_seqlen=2, device="cpu", dtype=torch.float32
    )

    first = router.step(torch.tensor([[[1.0, 0.0]]]), state)
    continuation = router.step(
        torch.tensor([[[-1.0, 0.0]]]),
        state,
        continuation_mask=torch.tensor([[True]]),
        continuation_hard=True,
    )

    assert first.boundary_mask.tolist() == [True]
    assert continuation.boundary_mask.tolist() == [False]
    assert continuation.boundary_prob[0].tolist() == [1.0, 0.0]


def test_recurrent_hard_constraint_preserves_sequence_start_boundary() -> None:
    router = RoutingModule(d_model=2, device="cpu", dtype=torch.float32)
    state = router.allocate_inference_cache(
        batch_size=1, max_seqlen=1, device="cpu", dtype=torch.float32
    )

    first = router.step(
        torch.tensor([[[1.0, 0.0]]]),
        state,
        continuation_mask=torch.tensor([[True]]),
        continuation_hard=True,
    )

    assert first.boundary_mask.tolist() == [True]


def test_hard_constraint_preserves_sequence_start_boundary() -> None:
    router = RoutingModule(d_model=2, device="cpu", dtype=torch.float32)
    hidden_states = torch.tensor(
        [[[1.0, 0.0], [-1.0, 0.0]]], dtype=torch.float32
    )
    mask = torch.ones((1, 2), dtype=torch.bool)

    output = router(
        hidden_states,
        mask=mask,
        continuation_mask=torch.tensor([[True, True]]),
        continuation_hard=True,
    )

    assert output.boundary_mask.tolist() == [[True, False]]


def test_boundary_override_replaces_mask_and_probabilities() -> None:
    router = RoutingModule(d_model=2, device="cpu", dtype=torch.float32)
    output = router(
        torch.tensor([[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]]),
        mask=torch.ones((1, 3), dtype=torch.bool),
    )
    probabilities = torch.tensor(
        [[[0.0, 1.0], [0.2, 0.8], [0.7, 0.3]]], dtype=torch.float32
    )

    overridden = apply_boundary_override(
        output,
        BoundaryOverride(
            boundary_mask=torch.tensor([[True, True, False]]),
            boundary_prob=probabilities,
        ),
    )

    assert overridden.boundary_mask.tolist() == [[True, True, False]]
    assert torch.equal(overridden.boundary_prob, probabilities)
    torch.testing.assert_close(
        overridden.selected_probs.squeeze(-1),
        torch.tensor([[1.0, 0.8, 0.7]]),
    )


def test_boundary_override_rejects_invalid_shape() -> None:
    router = RoutingModule(d_model=2, device="cpu", dtype=torch.float32)
    output = router(
        torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]]),
        mask=torch.ones((1, 2), dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="shape"):
        apply_boundary_override(
            output,
            BoundaryOverride(boundary_mask=torch.tensor([[True]])),
        )


def test_boundary_override_rejects_removed_sequence_start() -> None:
    router = RoutingModule(d_model=2, device="cpu", dtype=torch.float32)
    output = router(
        torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]]),
        mask=torch.ones((1, 2), dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="sequence start"):
        apply_boundary_override(
            output,
            BoundaryOverride(boundary_mask=torch.tensor([[False, True]])),
            required_start_mask=torch.tensor([[True, False]]),
        )
