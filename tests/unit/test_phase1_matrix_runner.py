import argparse

from scripts.run_phase1_matrix import parse_run


def test_parse_run_accepts_phase1_candidates() -> None:
    assert parse_run("k1g1:44") == ("k1g1", 44)
    assert parse_run("t26:42") == ("t26", 42)


def test_parse_run_rejects_unknown_candidates() -> None:
    try:
        parse_run("unknown:42")
    except argparse.ArgumentTypeError as exc:
        assert "unknown main network" in str(exc)
    else:
        raise AssertionError("unknown candidates must be rejected")
