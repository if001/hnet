from scripts.run_matched_boundary_training import run_name


def test_run_name_records_all_matched_training_controls() -> None:
    assert run_name("m3t1", 0.05, 3.5, 3.0, 42, "1234567890") == (
        "r5_match_m3t1_comp3p5-3_rw0p05_utf8hard_s42_step220_1234567"
    )


def test_run_name_records_seed_43() -> None:
    assert run_name("k1t1", 0.08, 3.0, 2.5, 43, "abcdef0123") == (
        "r5_match_k1t1_comp3-2p5_rw0p08_utf8hard_s43_step220_abcdef0"
    )


def test_run_name_records_seed_44() -> None:
    assert run_name("k1g1", 0.08, 3.0, 2.5, 44, "abcdef0123") == (
        "r5_match_k1g1_comp3-2p5_rw0p08_utf8hard_s44_step220_abcdef0"
    )
