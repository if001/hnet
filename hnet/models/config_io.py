import json
from dataclasses import asdict
from pathlib import Path

from .config_hnet import (
    AttnConfig,
    BoundaryFeatureConfig,
    FFNMoEConfig,
    HNetConfig,
    KDAConfig,
    MLAConfig,
    SSMConfig,
)


def load_hnet_config(config_path: str | Path) -> HNetConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    attn_cfg = AttnConfig(**config.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config.pop("ssm_cfg"))
    kda_cfg = KDAConfig(**config.pop("kda_cfg", {}))
    mla_cfg = MLAConfig(**config.pop("mla_cfg", {}))
    boundary_feature_cfg = BoundaryFeatureConfig(
        **config.pop("boundary_feature_cfg", {})
    )
    ffn_moe_cfg = FFNMoEConfig(**config.pop("ffn_moe_cfg", {}))
    return HNetConfig(
        **config,
        attn_cfg=attn_cfg,
        ssm_cfg=ssm_cfg,
        kda_cfg=kda_cfg,
        mla_cfg=mla_cfg,
        boundary_feature_cfg=boundary_feature_cfg,
        ffn_moe_cfg=ffn_moe_cfg,
    )


def save_hnet_config(config: HNetConfig, config_path: str | Path) -> Path:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, ensure_ascii=False, indent=4)
        handle.write("\n")
    return path
