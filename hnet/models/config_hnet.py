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
class MLAConfig:
    """Configuration for Kimi K3-style gated Multi-Latent Attention."""

    num_heads: List = field(default_factory=list)
    q_lora_rank: Any = None
    kv_lora_rank: Union[int, List] = 512
    qk_nope_head_dim: Union[int, List] = 128
    qk_rope_head_dim: Union[int, List] = 0
    v_head_dim: Union[int, List] = 128
    use_output_gate: bool = True


@dataclass
class BoundaryFeatureConfig:
    """Controls which encoder representations are sent to boundary routing."""

    mode: str = "final"
    final_logit_bias: float = 2.0


@dataclass
class FFNMoEConfig:
    """Top-1 expert routing for innermost main-network FFNs."""

    enabled: bool = False
    num_experts: int = 4
    top_k: int = 1
    capacity_factor: float = 1.25


@dataclass
class MixerMoEConfig:
    """Top-1 mixer routing at selected innermost main-network layers."""

    enabled: bool = False
    layer_indices: List[int] = field(default_factory=list)
    expert_arches: List[str] = field(default_factory=lambda: ["T", "K", "G"])
    top_k: int = 1
    initial_expert: str = "T"
    initial_bias: float = 2.0


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
    mla_cfg: MLAConfig = field(default_factory=MLAConfig)
    boundary_feature_cfg: BoundaryFeatureConfig = field(
        default_factory=BoundaryFeatureConfig
    )
    ffn_moe_cfg: FFNMoEConfig = field(default_factory=FFNMoEConfig)
    mixer_moe_cfg: MixerMoEConfig = field(default_factory=MixerMoEConfig)
    tie_embeddings: bool = False

    def to_dict(self):
        return asdict(self)
