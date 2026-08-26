import pytest
import torch

from hnet.modules.boundary_features import LayerScalarMix


def test_layer_scalar_mix_starts_with_final_layer_bias() -> None:
    mixer = LayerScalarMix(4, final_logit_bias=2.0)
    weights = mixer.normalized_weights()

    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert weights[-1] > weights[0]
    assert torch.allclose(weights[:3], weights[0].expand(3))


def test_layer_scalar_mix_combines_states_and_backpropagates() -> None:
    mixer = LayerScalarMix(2, final_logit_bias=0.0)
    first = torch.tensor([[1.0, 3.0]], requires_grad=True)
    second = torch.tensor([[3.0, 5.0]], requires_grad=True)

    output = mixer([first, second])
    assert torch.allclose(output, torch.tensor([[2.0, 4.0]]))

    output.sum().backward()
    assert first.grad is not None
    assert second.grad is not None
    assert mixer.logits.grad is not None


def test_layer_scalar_mix_validates_state_count_and_shape() -> None:
    mixer = LayerScalarMix(2)
    with pytest.raises(ValueError, match="count"):
        mixer([torch.zeros(1, 2)])
    with pytest.raises(ValueError, match="same shape"):
        mixer([torch.zeros(1, 2), torch.zeros(2, 2)])
