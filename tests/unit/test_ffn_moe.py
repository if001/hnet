import torch

from hnet.modules.mlp import Top1SwiGLUMoE


def test_top1_moe_reports_capacity_drop_and_router_gradient() -> None:
    module = Top1SwiGLUMoE(
        8,
        d_intermediate=16,
        multiple_of=1,
        num_experts=4,
        capacity_factor=1.25,
    )
    torch.nn.init.zeros_(module.router.weight)
    inputs = torch.randn(8, 8, requires_grad=True)

    output = module(inputs)
    loss = output.square().mean() + 0.01 * module.last_aux_loss
    loss.backward()

    assert output.shape == inputs.shape
    assert torch.isclose(module.last_assignment_fraction.max(), torch.tensor(1.0))
    assert torch.isclose(module.last_dropped_fraction, torch.tensor(5 / 8))
    assert module.last_routing_entropy > 0.99
    assert module.router.weight.grad is not None
    assert module.router.weight.grad.abs().sum() > 0


def test_top1_moe_rejects_non_top1_configuration() -> None:
    try:
        Top1SwiGLUMoE(8, num_experts=4, top_k=2)
    except ValueError as error:
        assert "top_k=1" in str(error)
    else:
        raise AssertionError("top_k=2 should be rejected")
