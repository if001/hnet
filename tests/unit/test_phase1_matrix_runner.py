import argparse

from scripts.run_phase1_matrix import CONDITIONS, parse_run


def test_parse_run_accepts_phase1_candidates() -> None:
    assert parse_run("k1g1:44") == ("k1g1", 44)
    assert parse_run("k3g1:42") == ("k3g1", 42)
    assert parse_run("k3t1:43") == ("k3t1", 43)
    assert parse_run("t26:42") == ("t26", 42)


def test_parse_run_rejects_unknown_candidates() -> None:
    try:
        parse_run("unknown:42")
    except argparse.ArgumentTypeError as exc:
        assert "unknown main network" in str(exc)
    else:
        raise AssertionError("unknown candidates must be rejected")


def test_k3t1_uses_calibrated_ratio_weight() -> None:
    assert CONDITIONS["k3t1"] == {
        "ratio_weight": 0.08,
        "inner": 3.0,
        "outer": 3.0,
    }
