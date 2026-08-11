import json

from hnet.models.config_hnet import KDAConfig, MLAConfig
from hnet.models.config_io import load_hnet_config, save_hnet_config


def test_kda_config_round_trip(tmp_path):
    source = tmp_path / "model.json"
    source.write_text(
        json.dumps(
            {
                "arch_layout": ["m1", ["T1", ["K1T1"], "T1"], "m1"],
                "d_model": [8, 8, 8],
                "d_intermediate": [0, 16, 16],
                "ssm_cfg": {},
                "attn_cfg": {"num_heads": [1, 1, 1]},
                "kda_cfg": {
                    "num_heads": [1, 1, 1],
                    "head_dim": [8, 8, 8],
                    "short_conv_kernel_size": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_hnet_config(source)
    assert config.kda_cfg == KDAConfig(
        num_heads=[1, 1, 1], head_dim=[8, 8, 8], short_conv_kernel_size=3
    )

    output = save_hnet_config(config, tmp_path / "saved.json")
    assert load_hnet_config(output) == config


def test_old_config_gets_empty_kda_config(tmp_path):
    source = tmp_path / "old.json"
    source.write_text(
        json.dumps(
            {
                "arch_layout": ["T1"],
                "d_model": [8],
                "d_intermediate": [16],
                "ssm_cfg": {},
                "attn_cfg": {"num_heads": [1]},
            }
        ),
        encoding="utf-8",
    )

    assert load_hnet_config(source).kda_cfg == KDAConfig()
    assert load_hnet_config(source).mla_cfg == MLAConfig()


def test_mla_config_round_trip(tmp_path):
    source = tmp_path / "model.json"
    source.write_text(
        json.dumps(
            {
                "arch_layout": ["K3G1"],
                "d_model": [64],
                "d_intermediate": [128],
                "ssm_cfg": {},
                "attn_cfg": {"num_heads": [4]},
                "mla_cfg": {
                    "num_heads": [4],
                    "q_lora_rank": [32],
                    "kv_lora_rank": [16],
                    "qk_nope_head_dim": [12],
                    "qk_rope_head_dim": [4],
                    "v_head_dim": [12],
                    "use_output_gate": True,
                },
            },
        ),
        encoding="utf-8",
    )
    config = load_hnet_config(source)
    assert config.mla_cfg == MLAConfig(
        num_heads=[4], q_lora_rank=[32], kv_lora_rank=[16],
        qk_nope_head_dim=[12], qk_rope_head_dim=[4], v_head_dim=[12],
        use_output_gate=True,
    )
    saved = save_hnet_config(config, tmp_path / "saved.json")
    assert load_hnet_config(saved) == config
