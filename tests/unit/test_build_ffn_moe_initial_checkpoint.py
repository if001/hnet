from scripts.build_ffn_moe_initial_checkpoint import dense_key_for_moe_expert


def test_dense_key_for_moe_expert_maps_each_expert_to_dense_ffn() -> None:
    assert dense_key_for_moe_expert(
        "backbone.main_network.main_network.main_network.layers.0."
        "mlp.experts.3.fc1.weight"
    ) == (
        "backbone.main_network.main_network.main_network.layers.0."
        "mlp.fc1.weight"
    )
    assert dense_key_for_moe_expert("backbone.encoder.layers.0.mlp.fc1.weight") is None
