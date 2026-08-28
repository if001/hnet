from dataclasses import dataclass

import torch


FREEZE_MODES = ("none", "router", "main", "outer")


@dataclass(frozen=True)
class FreezeSummary:
    mode: str
    trainable_parameters: int
    frozen_parameters: int


def is_router_parameter(name: str) -> bool:
    return "routing_module" in name.split(".")


def innermost_main_parameter_ids(model: torch.nn.Module) -> set[int]:
    """Return parameters belonging to the innermost language-model mixer."""
    parameter_ids: set[int] = set()
    for module in model.modules():
        if getattr(module, "is_innermost", False) and hasattr(module, "main_network"):
            parameter_ids.update(
                id(parameter) for parameter in module.main_network.parameters()
            )
    return parameter_ids


def apply_freeze_mode(
    model: torch.nn.Module,
    mode: str,
) -> FreezeSummary:
    """Select trainable parameters for router/main-network ablations."""
    if mode not in FREEZE_MODES:
        raise ValueError(f"Unsupported freeze mode: {mode}")

    trainable = 0
    frozen = 0
    innermost_ids = innermost_main_parameter_ids(model) if mode == "outer" else set()
    for name, parameter in model.named_parameters():
        router_parameter = is_router_parameter(name)
        if mode == "none":
            requires_grad = True
        elif mode == "router":
            requires_grad = not router_parameter
        elif mode == "main":
            requires_grad = router_parameter
        else:
            requires_grad = (
                id(parameter) not in innermost_ids
                and not name.startswith("embeddings.")
                and not name.startswith("lm_head.")
            )
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
