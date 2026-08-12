import pytest

from hnet.training.chunk_analysis import (
    mid_utf8_boundary_rate,
    percentile,
    summarize_lengths,
    target_boundary_offsets,
)


def test_summarize_lengths_reports_requested_percentiles() -> None:
    chunks = [[0], [0, 1], [0, 1, 2], list(range(10))]

    summary = summarize_lengths(chunks)

    assert summary["count"] == 4
    assert summary["mean"] == 4
    assert summary["median"] == 2.5
    assert summary["p50"] == percentile([1, 2, 3, 10], 0.5)
    assert summary["p90"] == pytest.approx(7.9)
    assert summary["p95"] == pytest.approx(8.95)
    assert summary["max"] == 10


def test_mid_utf8_boundary_rate_uses_continuation_bytes_as_candidates() -> None:
    token_ids = [254, 0xE7, 0x8C, 0xAB, ord("a")]

    assert mid_utf8_boundary_rate(token_ids, [0, 2]) == 0.5
    assert mid_utf8_boundary_rate(token_ids, [0, 1, 4]) == 0.0


def test_target_boundary_offsets_account_for_bos() -> None:
    prompt = "猫が笑っている。笑っている！"
    target = "笑っている"
    first_start = len("猫が".encode("utf-8")) + 1
    target_length = len(target.encode("utf-8"))
    second_start = first_start + target_length + len("。".encode("utf-8"))

    offsets = target_boundary_offsets(
        prompt,
        target,
        [first_start, first_start + 3, second_start + 6],
    )

    assert offsets == [[0, 3], [6]]
