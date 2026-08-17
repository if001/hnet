import math

import pytest

from scripts.evaluate_text_bpb import aggregate_rows, bootstrap_paired_mean


def test_aggregate_rows_uses_raw_byte_micro_average():
    rows = [
        {"category": "ja", "raw_bytes": 10, "nll_sum": math.log(2) * 10, "bpb": 1.0},
        {"category": "ja", "raw_bytes": 30, "nll_sum": math.log(2) * 60, "bpb": 2.0},
    ]
    summary = {row["category"]: row for row in aggregate_rows(rows)}
    assert summary["ja"]["micro_bpb"] == pytest.approx(1.75)
    assert summary["ja"]["record_macro_bpb"] == pytest.approx(1.5)
    assert summary["__overall__"]["raw_bytes"] == 40


def test_bootstrap_interval_is_deterministic():
    assert bootstrap_paired_mean([1.0, 2.0], samples=100) == bootstrap_paired_mean(
        [1.0, 2.0], samples=100
    )
