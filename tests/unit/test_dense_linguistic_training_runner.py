from scripts.run_dense_linguistic_training import DENSE_STEPS, run_name


def test_dense_run_name_records_main_seed_and_commit() -> None:
    assert run_name("k1g1", 42, "abcdef0123") == (
        "r6_dense_family_v1_k1g1_s42_step220_abcdef0"
    )


def test_dense_steps_are_ten_step_intervals() -> None:
    assert DENSE_STEPS == tuple(range(10, 221, 10))
