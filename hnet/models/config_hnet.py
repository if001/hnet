from dataclasses import dataclass, field, asdict
from typing import Any, List, Union


@dataclass
class AttnConfig:
    num_heads: List = field(default_factory=list)
    rotary_emb_dim: List = field(default_factory=list)
    window_size: List = field(default_factory=list)
    rope_scaling: Any = None


@dataclass
class SSMConfig:
    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    chunk_size: int = 256


@dataclass
class KDAConfig:
    """Configuration for Kimi Delta Attention layers."""

    num_heads: List = field(default_factory=list)
    head_dim: List = field(default_factory=list)
    short_conv_kernel_size: Union[int, List] = 4
    use_full_rank_gate: bool = False
    gate_lower_bound: Any = None


@dataclass
class HNetConfig:
    arch_layout: List[Union[str, List]] = field(default_factory=list)
    d_model: List[int] = field(default_factory=list)
    # intermediate dimension for the FFNs (0 indicates no FFN)
    d_intermediate: List[int] = field(default_factory=list)
    vocab_size: int = 256
    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    kda_cfg: KDAConfig = field(default_factory=KDAConfig)
    tie_embeddings: bool = False

    def to_dict(self):
        return asdict(self)
