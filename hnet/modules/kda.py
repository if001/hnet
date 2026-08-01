"""H-Net adapter for Kimi K3's Kimi Delta Attention implementation."""

from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange
from torch import nn

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
except ImportError as exc:  # pragma: no cover - depends on the CUDA environment
    raise ImportError(
        "Kimi Delta Attention layers require fla-core; install it with "
        "`pip install -U fla-core`."
    ) from exc


class KimiDeltaAttention(nn.Module):
    """Kimi K3 delta attention with H-Net's mixer/cache interface.

    Unlike H-Net MHA, KDA is a causal recurrent linear attention mechanism and
    deliberately has no RoPE. Sequence order is represented by its gated delta
    recurrence and causal short convolutions, matching ``kimi/modeling_kimi_linear.py``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        short_conv_kernel_size: int = 4,
        use_full_rank_gate: bool = False,
        gate_lower_bound: Optional[float] = None,
        layer_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if d_model != num_heads * head_dim:
            raise ValueError("KDA requires d_model == num_heads * head_dim")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.gate_lower_bound = gate_lower_bound
        self.use_full_rank_gate = use_full_rank_gate

        self.q_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.q_conv1d = ShortConvolution(
            d_model, short_conv_kernel_size, activation="silu", **factory_kwargs
        )
        self.k_conv1d = ShortConvolution(
            d_model, short_conv_kernel_size, activation="silu", **factory_kwargs
        )
        self.v_conv1d = ShortConvolution(
            d_model, short_conv_kernel_size, activation="silu", **factory_kwargs
        )
        self.A_log = nn.Parameter(
            torch.empty(num_heads, device=device, dtype=torch.float32)
        )
        nn.init.uniform_(self.A_log, 1, 16)
        with torch.no_grad():
            self.A_log.log_()
        self.f_a_proj = nn.Linear(d_model, head_dim, bias=False, **factory_kwargs)
        self.f_b_proj = nn.Linear(head_dim, d_model, bias=False, **factory_kwargs)
        self.dt_bias = nn.Parameter(
            torch.empty(d_model, device=device, dtype=torch.float32)
        )
        nn.init.zeros_(self.dt_bias)
        self.b_proj = nn.Linear(d_model, num_heads, bias=False, **factory_kwargs)
        if use_full_rank_gate:
            self.g_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        else:
            self.g_a_proj = nn.Linear(d_model, head_dim, bias=False, **factory_kwargs)
            self.g_b_proj = nn.Linear(head_dim, d_model, bias=False, **factory_kwargs)
        self.o_norm = FusedRMSNormGated(head_dim, eps=1e-5, activation="sigmoid")
        self.o_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        del batch_size, max_seqlen, dtype, kwargs
        return {"conv": (None, None, None), "recurrent": None}

    def _cache(self, inference_params):
        if inference_params is None:
            return None
        return inference_params.key_value_memory_dict[self.layer_idx]

    def forward(
        self,
        hidden_states,
        inference_params=None,
        attention_mask=None,
        cu_seqlens=None,
        **kwargs,
    ):
        del kwargs
        if attention_mask is not None and not bool(torch.all(attention_mask)):
            raise ValueError("KDA currently supports packed input or an all-true mask")
        cache = self._cache(inference_params)
        conv_states = (None, None, None) if cache is None else cache["conv"]
        recurrent_state = None if cache is None else cache["recurrent"]
        use_cache = cache is not None

        projected = (self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states))
        convolutions = (self.q_conv1d, self.k_conv1d, self.v_conv1d)
        values = [
            conv(x=x, cache=state, output_final_state=use_cache, cu_seqlens=cu_seqlens)
            for conv, x, state in zip(convolutions, projected, conv_states)
        ]
        q, k, v = [rearrange(value[0], "... (h d) -> ... h d", d=self.head_dim) for value in values]
        delta_gate = rearrange(
            self.f_b_proj(self.f_a_proj(hidden_states)), "... (h d) -> ... h d", d=self.head_dim
        )
        beta = self.b_proj(hidden_states).float()
        kernel = fused_recurrent_kda if use_cache and hidden_states.shape[-2] == 1 else chunk_kda
        output, recurrent_state = kernel(
            q=q, k=k, v=v, g=delta_gate, beta=beta, A_log=self.A_log,
            dt_bias=self.dt_bias, initial_state=recurrent_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, lower_bound=self.gate_lower_bound,
            transpose_state_layout=True, cu_seqlens=cu_seqlens,
            **({"safe_gate": self.gate_lower_bound is not None} if kernel is chunk_kda else {}),
        )
        if cache is not None:
            cache["conv"] = tuple(value[1] for value in values)
            cache["recurrent"] = recurrent_state
        gate = self.g_proj(hidden_states) if self.use_full_rank_gate else self.g_b_proj(self.g_a_proj(hidden_states))
        gate = rearrange(gate, "... (h d) -> ... h d", d=self.head_dim)
        output = self.o_norm(output, gate)
        return self.o_proj(rearrange(output, "b t h d -> b t (h d)"))

    def step(self, hidden_states, inference_params):
        return self.forward(hidden_states, inference_params=inference_params)
