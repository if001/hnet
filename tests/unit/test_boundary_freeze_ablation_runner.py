from scripts.run_boundary_freeze_ablation import run_name


def test_run_name_records_scope_seed_steps_and_commit() -> None:
    assert run_name("k1t1", "router", 42, "1234567890") == (
        "r5_freeze_k1t1_router_s42_from165_to220_1234567"
    )
