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


class ToyHierarchicalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(4, 2)
        self.lm_head = torch.nn.Linear(2, 4)
        self.backbone = torch.nn.Module()
        self.backbone.is_innermost = False
        self.backbone.encoder = torch.nn.Linear(2, 2)
        self.backbone.routing_module = torch.nn.Linear(2, 1)
        self.backbone.dechunk_layer = torch.nn.Linear(2, 2)
        self.backbone.main_network = torch.nn.Module()
        self.backbone.main_network.is_innermost = True
        self.backbone.main_network.main_network = torch.nn.Linear(2, 2)
        self.backbone.main_network.pad_dimension = torch.nn.Parameter(torch.zeros(2))


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


def test_outer_mode_freezes_embeddings_head_and_innermost_mixer() -> None:
    model = ToyHierarchicalModel()
    summary = apply_freeze_mode(model, "outer")

    assert not model.embeddings.weight.requires_grad
    assert not model.lm_head.weight.requires_grad
    assert not model.backbone.main_network.main_network.weight.requires_grad
    assert model.backbone.encoder.weight.requires_grad
    assert model.backbone.routing_module.weight.requires_grad
    assert model.backbone.dechunk_layer.weight.requires_grad
    assert model.backbone.main_network.pad_dimension.requires_grad
    assert summary.trainable_parameters > 0
    assert summary.frozen_parameters > 0
