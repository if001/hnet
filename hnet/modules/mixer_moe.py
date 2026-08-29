from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch
from torch import nn

from .mha import CausalMHA
from .mla import GatedMLA


class Top1MixerMoE(nn.Module):
    """Top-1 output routing over full-sequence T, KDA, and Gated MLA experts.

    All experts evaluate the full causal sequence in this correctness pilot. The
    hard route selects one expert output per chunk position, while separate
    expert caches preserve autoregressive semantics. Sparse expert execution is
    intentionally deferred until routing quality has been established.
    """

    def __init__(
        self,
        d_model: int,
        *,
        attn_cfg: Mapping[str, object],
        kda_cfg: Mapping[str, object],
        mla_cfg: Mapping[str, object],
        expert_arches: list[str],
        top_k: int = 1,
        initial_expert: str = "T",
        initial_bias: float = 2.0,
        layer_idx: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if top_k != 1:
            raise ValueError("Top1MixerMoE currently requires top_k=1")
        if len(expert_arches) < 2 or len(set(expert_arches)) != len(expert_arches):
            raise ValueError("expert_arches must contain at least two unique mixers")
        if any(arch not in {"T", "K", "G"} for arch in expert_arches):
            raise ValueError("Mixer-MoE experts must be selected from T, K, and G")
        if initial_expert not in expert_arches:
            raise ValueError("initial_expert must be present in expert_arches")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.expert_arches = tuple(expert_arches)
        self.num_experts = len(expert_arches)
        self.router = nn.Linear(
            d_model, self.num_experts, bias=True, **factory_kwargs
        )
        with torch.no_grad():
            self.router.bias.zero_()
            self.router.bias[expert_arches.index(initial_expert)] = initial_bias

        experts = []
        for arch in expert_arches:
            if arch == "T":
                expert = CausalMHA(
                    d_model, **attn_cfg, layer_idx=layer_idx, **factory_kwargs
                )
            elif arch == "K":
                from .kda import KimiDeltaAttention

                expert = KimiDeltaAttention(
                    d_model, **kda_cfg, layer_idx=layer_idx, **factory_kwargs
                )
            else:
                expert = GatedMLA(
                    d_model, **mla_cfg, layer_idx=layer_idx, **factory_kwargs
                )
            experts.append(expert)
        self.experts = nn.ModuleList(experts)

        self.last_aux_loss: torch.Tensor | None = None
        self.last_assignment_fraction: torch.Tensor | None = None
        self.last_assignment_counts: torch.Tensor | None = None
        self.last_accepted_fraction: torch.Tensor | None = None
        self.last_accepted_counts: torch.Tensor | None = None
        self.last_dropped_fraction: torch.Tensor | None = None
        self.last_routing_entropy: torch.Tensor | None = None
        self.last_token_count: int = 0

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        return {
            "expert_caches": [
                expert.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                )
                for expert in self.experts
            ]
        }

    def _expert_inference_params(self, inference_params, expert_index: int):
        if inference_params is None:
            return None
        if self.layer_idx is None:
            raise ValueError("Mixer-MoE cached inference requires layer_idx")
        layer_cache = inference_params.key_value_memory_dict[self.layer_idx]
        proxy = copy.copy(inference_params)
        proxy.key_value_memory_dict = {
            self.layer_idx: layer_cache["expert_caches"][expert_index]
        }
        return proxy

    def _expert_forward(
        self,
        expert_index: int,
        hidden_states: torch.Tensor,
        *,
        inference_params=None,
        cu_seqlens=None,
        max_seqlen=None,
        attention_mask=None,
    ) -> torch.Tensor:
        arch = self.expert_arches[expert_index]
        expert = self.experts[expert_index]
        expert_inference = self._expert_inference_params(
            inference_params, expert_index
        )
        if arch == "K":
            packed_two_dimensional = (
                cu_seqlens is not None and hidden_states.dim() == 2
            )
            expert_input = (
                hidden_states.unsqueeze(0)
                if packed_two_dimensional
                else hidden_states
            )
            output = expert(
                expert_input,
                inference_params=expert_inference,
                attention_mask=attention_mask,
                cu_seqlens=cu_seqlens,
            )
            return output.squeeze(0) if packed_two_dimensional else output
        return expert(
            hidden_states,
            inference_params=expert_inference,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params=None,
        cu_seqlens=None,
        max_seqlen=None,
        attention_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        probabilities = torch.softmax(
            self.router(hidden_states).float(), dim=-1
        )
        chosen_probability, chosen_expert = probabilities.max(dim=-1)
        outputs = torch.stack(
            [
                self._expert_forward(
                    index,
                    hidden_states,
                    inference_params=inference_params,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    attention_mask=attention_mask,
                )
                for index in range(self.num_experts)
            ],
            dim=-2,
        )
        selected = outputs.gather(
            -2,
            chosen_expert.unsqueeze(-1).unsqueeze(-1).expand(
                *chosen_expert.shape, 1, hidden_states.shape[-1]
            ),
        ).squeeze(-2)
        routing_scale = (
            chosen_probability / chosen_probability.detach().clamp_min(1e-6)
        )
        selected = selected * routing_scale.to(selected.dtype).unsqueeze(-1)

        flat_choice = chosen_expert.reshape(-1)
        token_count = flat_choice.numel()
        counts = torch.bincount(flat_choice, minlength=self.num_experts)
        assignment_fraction = counts.float() / max(1, token_count)
        mean_probability = probabilities.reshape(-1, self.num_experts).mean(dim=0)
        self.last_aux_loss = self.num_experts * torch.sum(
            mean_probability * assignment_fraction
        )
        entropy = -(
            probabilities * probabilities.clamp_min(1e-9).log()
        ).sum(dim=-1)
        self.last_routing_entropy = entropy.mean() / math.log(self.num_experts)
        self.last_assignment_counts = counts.detach()
        self.last_assignment_fraction = assignment_fraction.detach()
        self.last_accepted_counts = counts.detach()
        self.last_accepted_fraction = assignment_fraction.detach()
        self.last_dropped_fraction = torch.zeros(
            (), device=hidden_states.device, dtype=torch.float32
        )
        self.last_token_count = token_count
        return selected

    def step(self, hidden_states, inference_params):
        return self.forward(hidden_states, inference_params=inference_params)
