# Base code imported from
# https://github.com/state-spaces/mamba
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn.ops.activations import swiglu


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model,
        d_intermediate=None,
        bias=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        d_intermediate = (
            d_intermediate if d_intermediate is not None else int(8 * d_model / 3)
        )
        d_intermediate = (d_intermediate + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(d_model, 2 * d_intermediate, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(d_intermediate, d_model, bias=bias, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = swiglu(gate, y) if y.is_cuda else F.silu(gate) * y
        y = self.fc2(y)
        return y


class Top1SwiGLUMoE(nn.Module):
    """Token-wise top-1 SwiGLU experts with a fixed per-expert capacity."""

    def __init__(
        self,
        d_model,
        d_intermediate=None,
        bias=False,
        multiple_of=128,
        num_experts=4,
        top_k=1,
        capacity_factor=1.25,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be at least 2")
        if top_k != 1:
            raise ValueError("Top1SwiGLUMoE currently requires top_k=1")
        if capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_experts = int(num_experts)
        self.capacity_factor = float(capacity_factor)
        self.router = nn.Linear(
            d_model, self.num_experts, bias=False, **factory_kwargs
        )
        self.experts = nn.ModuleList(
            SwiGLU(
                d_model,
                d_intermediate=d_intermediate,
                bias=bias,
                multiple_of=multiple_of,
                **factory_kwargs,
            )
            for _ in range(self.num_experts)
        )
        self.last_aux_loss: torch.Tensor | None = None
        self.last_assignment_fraction: torch.Tensor | None = None
        self.last_assignment_counts: torch.Tensor | None = None
        self.last_accepted_fraction: torch.Tensor | None = None
        self.last_accepted_counts: torch.Tensor | None = None
        self.last_dropped_fraction: torch.Tensor | None = None
        self.last_routing_entropy: torch.Tensor | None = None
        self.last_token_count: int = 0

    def forward(self, x):
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        probabilities = torch.softmax(self.router(flat).float(), dim=-1)
        chosen_probability, chosen_expert = probabilities.max(dim=-1)
        token_count = int(flat.shape[0])
        capacity = max(
            1,
            math.ceil(self.capacity_factor * token_count / self.num_experts),
        )
        output = torch.zeros_like(flat)
        accepted = torch.zeros(token_count, dtype=torch.bool, device=flat.device)
        assignment_counts = []
        accepted_counts = []

        for expert_index, expert in enumerate(self.experts):
            indices = torch.nonzero(
                chosen_expert == expert_index, as_tuple=False
            ).squeeze(-1)
            assignment_counts.append(indices.numel())
            if indices.numel() > capacity:
                scores = chosen_probability.index_select(0, indices)
                keep = torch.topk(scores, capacity, sorted=False).indices
                indices = indices.index_select(0, keep)
            accepted_counts.append(indices.numel())
            if indices.numel() == 0:
                continue
            expert_input = flat.index_select(0, indices)
            expert_output = expert(expert_input)
            selected_probability = chosen_probability.index_select(0, indices)
            routing_scale = (
                selected_probability
                / selected_probability.detach().clamp_min(1e-6)
            )
            output.index_copy_(
                0,
                indices,
                expert_output
                * routing_scale.to(dtype=expert_output.dtype).unsqueeze(-1),
            )
            accepted[indices] = True

        assignment_fraction = torch.tensor(
            assignment_counts, device=flat.device, dtype=torch.float32
        ) / max(1, token_count)
        accepted_fraction = torch.tensor(
            accepted_counts, device=flat.device, dtype=torch.float32
        ) / max(1, token_count)
        mean_probability = probabilities.mean(dim=0)
        self.last_aux_loss = self.num_experts * torch.sum(
            mean_probability * assignment_fraction
        )
        entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1)
        self.last_routing_entropy = entropy.mean() / math.log(self.num_experts)
        self.last_assignment_fraction = assignment_fraction.detach()
        self.last_assignment_counts = torch.tensor(
            assignment_counts, device=flat.device, dtype=torch.long
        )
        self.last_accepted_fraction = accepted_fraction.detach()
        self.last_accepted_counts = torch.tensor(
            accepted_counts, device=flat.device, dtype=torch.long
        )
        self.last_dropped_fraction = (~accepted).float().mean().detach()
        self.last_token_count = token_count
        return output.view(original_shape)
