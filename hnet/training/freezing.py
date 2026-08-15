from dataclasses import dataclass

import torch


FREEZE_MODES = ("none", "router", "main")


@dataclass(frozen=True)
class FreezeSummary:
    mode: str
    trainable_parameters: int
    frozen_parameters: int


def is_router_parameter(name: str) -> bool:
    return "routing_module" in name.split(".")


def apply_freeze_mode(
    model: torch.nn.Module,
    mode: str,
) -> FreezeSummary:
    """Select trainable parameters for router/main-network ablations."""
    if mode not in FREEZE_MODES:
        raise ValueError(f"Unsupported freeze mode: {mode}")

    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        router_parameter = is_router_parameter(name)
        if mode == "none":
            requires_grad = True
        elif mode == "router":
            requires_grad = not router_parameter
        else:
            requires_grad = router_parameter
        parameter.requires_grad_(requires_grad)
        if requires_grad:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()

    if trainable == 0:
        raise ValueError(f"Freeze mode {mode!r} left no trainable parameters")
    return FreezeSummary(
        mode=mode,
        trainable_parameters=trainable,
        frozen_parameters=frozen,
    )
