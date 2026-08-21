from scripts.run_dense_linguistic_training import (
    DENSE_STEPS,
    OUTER_COMPRESSION_TARGETS,
    checkpoint_steps,
    dense_steps,
    run_name,
)


def test_dense_run_name_records_main_seed_and_commit() -> None:
    assert run_name("k1g1", 42, "abcdef0123") == (
        "r6_dense_family_v1_k1g1_s42_step220_abcdef0"
    )


def test_dense_steps_are_ten_step_intervals() -> None:
    assert DENSE_STEPS == tuple(range(10, 221, 10))
    assert dense_steps(55) == (10, 20, 30, 40, 50)
    assert checkpoint_steps(55) == (55,)


def test_dense_targets_include_combined_pareto_candidates() -> None:
    assert OUTER_COMPRESSION_TARGETS == {
        "k1g1": 2.5,
        "k3g1": 2.5,
        "k3t1": 3.0,
        "k1first_mix": 2.5,
        "k3first_mix": 2.5,
    }


def test_dense_run_name_supports_screening_length() -> None:
    assert run_name("k1first_mix", 42, "abcdef0123", 55) == (
        "r6_dense_family_v1_k1first_mix_s42_step55_abcdef0"
    )
