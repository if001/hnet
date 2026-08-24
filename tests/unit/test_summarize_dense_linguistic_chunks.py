import pytest

from scripts.summarize_dense_linguistic_chunks import align_probe_observations


def test_align_probe_observations_reuses_text_for_multiple_annotations() -> None:
    probe_records = [
        {"id": "category", "text": "同じ文章。"},
        {"id": "family", "text": "同じ文章。"},
        {"id": "other", "text": "別の文章。"},
    ]
    report_records = [
        {"text": "同じ文章。", "stage0": {}},
        {"text": "別の文章。", "stage0": {}},
    ]

    aligned = align_probe_observations(probe_records, report_records)

    assert [annotation["id"] for annotation, _ in aligned] == [
        "category",
        "family",
        "other",
    ]
    assert aligned[0][1] is aligned[1][1]


def test_align_probe_observations_rejects_missing_or_duplicate_reports() -> None:
    probe_records = [{"id": "one", "text": "文章。"}]
    with pytest.raises(ValueError, match="mismatch"):
        align_probe_observations(probe_records, [])
    with pytest.raises(ValueError, match="must be unique"):
        align_probe_observations(
            probe_records,
            [{"text": "文章。"}, {"text": "文章。"}],
        )
