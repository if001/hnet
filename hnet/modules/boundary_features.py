from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class LayerScalarMix(nn.Module):
    """Learn a global convex mixture over encoder-layer representations."""

    def __init__(
        self,
        num_layers: int,
        *,
        final_logit_bias: float = 2.0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        logits = torch.zeros(num_layers, device=device, dtype=torch.float32)
        logits[-1] = float(final_logit_bias)
        self.logits = nn.Parameter(logits)

    def normalized_weights(self) -> torch.Tensor:
        return torch.softmax(self.logits.float(), dim=0)

    def forward(self, layer_states: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(layer_states) != self.logits.numel():
            raise ValueError(
                "Layer-state count does not match scalar-mix weights: "
                f"{len(layer_states)} != {self.logits.numel()}"
            )
        reference_shape = layer_states[0].shape
        if any(state.shape != reference_shape for state in layer_states[1:]):
            raise ValueError("All layer states must have the same shape")
        stacked = torch.stack(tuple(layer_states), dim=0)
        weights = self.normalized_weights().to(
            device=stacked.device, dtype=stacked.dtype
        )
        weights = weights.reshape((weights.shape[0],) + (1,) * (stacked.dim() - 1))
        return torch.sum(stacked * weights, dim=0)
