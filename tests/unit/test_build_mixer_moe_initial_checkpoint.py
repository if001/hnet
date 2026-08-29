from scripts.build_mixer_moe_initial_checkpoint import (
    dense_key_for_transformer_expert,
)


def test_dense_key_for_transformer_expert() -> None:
    key = "backbone.main.layers.11.mixer.experts.0.Wqkv.weight"
    assert dense_key_for_transformer_expert(key) == (
        "backbone.main.layers.11.mixer.Wqkv.weight"
    )
    assert dense_key_for_transformer_expert(
        "backbone.main.layers.11.mixer.experts.1.q_proj.weight"
    ) is None
