import pytest
import torch

from hnet.training.freezing import apply_freeze_mode, is_router_parameter


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = torch.nn.Linear(2, 2)
        self.backbone = torch.nn.Module()
        self.backbone.routing_module = torch.nn.Linear(2, 1)
        self.backbone.main_network = torch.nn.Linear(2, 2)


def test_is_router_parameter_matches_module_path_segment() -> None:
    assert is_router_parameter("backbone.routing_module.weight")
    assert not is_router_parameter("backbone.main_network.weight")
    assert not is_router_parameter("backbone.routing_module_backup.weight")


@pytest.mark.parametrize(
    ("mode", "router_trainable", "main_trainable"),
    [
        ("none", True, True),
        ("router", False, True),
        ("main", True, False),
    ],
)
def test_apply_freeze_mode_selects_expected_parameters(
    mode: str,
    router_trainable: bool,
    main_trainable: bool,
) -> None:
    model = ToyModel()
    summary = apply_freeze_mode(model, mode)

    assert model.backbone.routing_module.weight.requires_grad is router_trainable
    assert model.backbone.main_network.weight.requires_grad is main_trainable
    assert model.embeddings.weight.requires_grad is main_trainable
    assert summary.trainable_parameters > 0


def test_apply_freeze_mode_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported freeze mode"):
        apply_freeze_mode(ToyModel(), "unknown")
