"""Kimi K3-style gated Multi-Latent Attention for H-Net."""

import torch
from einops import rearrange
from torch import nn

from .mha import FlashCausalCrossAttention, _update_kv_cache


class LatentRMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-6, **factory_kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension, **factory_kwargs))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        x_float *= torch.rsqrt(x_float.square().mean(-1, keepdim=True) + self.eps)
        return x_float.to(x.dtype) * self.weight


class GatedMLA(nn.Module):
    """MLA with Kimi K3's per-head sigmoid output gate.

    K3 uses MLA in NoPE mode. Its historically named ``qk_rope_head_dim`` is
    retained as an unrotated, shared-key subspace; no positional transform is
    applied here.
    """

    def __init__(
        self, d_model: int, num_heads: int, kv_lora_rank: int,
        qk_nope_head_dim: int, v_head_dim: int, qk_rope_head_dim: int = 0,
        q_lora_rank: int | None = None, use_output_gate: bool = True,
        layer_idx: int | None = None, device=None, dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        if v_head_dim > self.q_head_dim:
            raise ValueError("v_head_dim must not exceed the total query head dimension")

        self.q_proj = self.q_a_proj = self.q_a_norm = self.q_b_proj = None
        if q_lora_rank is None:
            self.q_proj = nn.Linear(d_model, num_heads * self.q_head_dim,
                                    bias=False, **factory_kwargs)
        else:
            self.q_a_proj = nn.Linear(d_model, q_lora_rank, bias=False,
                                      **factory_kwargs)
            self.q_a_norm = LatentRMSNorm(q_lora_rank, **factory_kwargs)
            self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.q_head_dim,
                                      bias=False, **factory_kwargs)
        self.kv_a_proj = nn.Linear(d_model, kv_lora_rank + qk_rope_head_dim,
                                   bias=False, **factory_kwargs)
        self.kv_a_norm = LatentRMSNorm(kv_lora_rank, **factory_kwargs)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False, **factory_kwargs)
        self.gate_proj = (nn.Linear(d_model, num_heads * v_head_dim, bias=False,
                                    **factory_kwargs) if use_output_gate else None)
        self.out_proj = nn.Linear(num_heads * v_head_dim, d_model, bias=False,
                                  **factory_kwargs)
        self.inner_attn = FlashCausalCrossAttention(window_size=-1)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        return torch.empty(batch_size, max_seqlen, 2, self.num_heads,
                           self.q_head_dim, device=self.out_proj.weight.device,
                           dtype=dtype)

    def forward(self, x: torch.Tensor, cu_seqlens=None, max_seqlen=None,
                inference_params=None, **kwargs) -> torch.Tensor:
        del kwargs
        if self.q_proj is not None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        q = rearrange(q, "... (h d) -> ... h d", h=self.num_heads)
        latent, shared_key = torch.split(
            self.kv_a_proj(x), [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = rearrange(
            self.kv_b_proj(self.kv_a_norm(latent)), "... (h d) -> ... h d",
            h=self.num_heads, d=self.qk_nope_head_dim + self.v_head_dim)
        key, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], -1)
        if self.qk_rope_head_dim:
            shared_key = shared_key.unsqueeze(-2).expand(*key.shape[:-1], -1)
            key = torch.cat((key, shared_key), -1)
        value = torch.nn.functional.pad(value, (0, self.q_head_dim - self.v_head_dim))
        packed_kv = torch.stack((key, value), dim=-3)

        if inference_params is not None:
            if cu_seqlens is not None:
                raise ValueError("packed MLA cannot be combined with inference cache")
            packed_kv = _update_kv_cache(packed_kv, inference_params, self.layer_idx)
            context = self.inner_attn(q, packed_kv)
        elif cu_seqlens is not None:
            context = self.inner_attn(
                q, packed_kv, cu_seqlens=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen=max_seqlen, max_seqlen_k=max_seqlen)
        else:
            context = self.inner_attn(q, packed_kv)
        context = rearrange(context[..., :self.v_head_dim], "... h d -> ... (h d)")
        if self.gate_proj is not None:
            context = context * self.gate_proj(x).sigmoid()
        return self.out_proj(context)

    def step(self, x: torch.Tensor, inference_params) -> torch.Tensor:
        return self.forward(x, inference_params=inference_params)
