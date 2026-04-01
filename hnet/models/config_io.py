import json
from pathlib import Path

from .config_hnet import AttnConfig, HNetConfig, SSMConfig


def load_hnet_config(config_path: str | Path) -> HNetConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    attn_cfg = AttnConfig(**config.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config.pop("ssm_cfg"))
    return HNetConfig(**config, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
