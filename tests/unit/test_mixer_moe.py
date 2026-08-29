import math

import torch
from torch import nn

from hnet.modules.mixer_moe import Top1MixerMoE


class ScaleExpert(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, inputs, **kwargs):
        del kwargs
        return inputs * self.scale


def test_top1_mixer_moe_selects_per_position_and_trains_router() -> None:
    module = Top1MixerMoE.__new__(Top1MixerMoE)
    nn.Module.__init__(module)
    module.d_model = 2
    module.layer_idx = 0
    module.expert_arches = ("T", "K", "G")
    module.num_experts = 3
    module.router = nn.Linear(2, 3, bias=False)
    module.experts = nn.ModuleList(
        [ScaleExpert(1.0), ScaleExpert(2.0), ScaleExpert(3.0)]
    )
    with torch.no_grad():
        module.router.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
        )

    inputs = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], requires_grad=True)
    output = module(inputs)
    expected = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]])
    assert torch.allclose(output, expected)
    assert module.last_assignment_counts.tolist() == [1, 1, 0]
    assert module.last_dropped_fraction.item() == 0.0
    assert 0.0 < module.last_routing_entropy.item() < 1.0

    loss = output.square().mean() + 0.01 * module.last_aux_loss
    loss.backward()
    assert module.router.weight.grad is not None
    assert module.router.weight.grad.abs().sum() > 0
    assert math.isfinite(module.last_aux_loss.item())
