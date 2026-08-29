import json

import numpy as np

from hnet.training.data import (
    DefaultRecordFormatter,
    PackedCurriculumByteDataset,
    PackedMixByteDataset,
)


def test_default_record_formatter_prefers_known_text_fields() -> None:
    formatter = DefaultRecordFormatter()

    record = {
        "instruction": "Explain Japanese grammar.",
        "output": "This is the answer.",
        "ignored": 123,
    }

    assert formatter.format_record(record) == "This is the answer.\n\nExplain Japanese grammar."


def test_default_record_formatter_handles_message_lists() -> None:
    formatter = DefaultRecordFormatter()

    record = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    }

    assert formatter.format_record(record) == "user: Hi\nassistant: Hello"


def test_packed_dataset_supports_uint32_tokens_and_raw_byte_lengths(tmp_path) -> None:
    source_dir = tmp_path / "datasets" / "source"
    source_dir.mkdir(parents=True)
    np.asarray([1, 4, 5, 2], dtype=np.uint32).tofile(source_dir / "data-00000.bin")
    (source_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dtype": "uint32",
                "shards": [
                    {
                        "bin_file": "data-00000.bin",
                        "token_count": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "mix_manifest.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "source",
                        "manifest": "datasets/source/manifest.json",
                    }
                ],
                "token_byte_lengths": [0, 0, 0, 0, 3, 1],
                "is_byte_level": False,
            }
        ),
        encoding="utf-8",
    )

    dataset = PackedMixByteDataset(tmp_path, seq_len=3, shuffle=False)
    sample = dataset[0]

    assert sample["input_ids"].tolist() == [1, 4, 5]
    assert sample["labels"].tolist() == [4, 5, 2]
    assert sample["target_byte_lengths"].tolist() == [3, 1, 0]
    assert not bool(sample["is_byte_level"])


def test_packed_dataset_order_audit_is_seed_specific_and_reproducible(tmp_path) -> None:
    source_dir = tmp_path / "datasets" / "source"
    source_dir.mkdir(parents=True)
    np.arange(40, dtype=np.uint8).tofile(source_dir / "data-00000.bin")
    (source_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dtype": "uint8",
                "shards": [{"bin_file": "data-00000.bin", "token_count": 40}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "mix_manifest.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {"name": "source", "manifest": "datasets/source/manifest.json"}
                ],
                "is_byte_level": True,
            }
        ),
        encoding="utf-8",
    )

    first = PackedMixByteDataset(tmp_path, seq_len=3, shuffle=True, seed=42)
    repeated = PackedMixByteDataset(tmp_path, seq_len=3, shuffle=True, seed=42)
    changed = PackedMixByteDataset(tmp_path, seq_len=3, shuffle=True, seed=43)

    first_audit = first.order_audit(sample_limit=4)
    assert first_audit == repeated.order_audit(sample_limit=4)
    assert first_audit["shuffle_index_sha256"] != changed.order_audit()[
        "shuffle_index_sha256"
    ]
    assert len(first_audit["sample_prefix"]) == 4


def test_packed_curriculum_aligns_short_contexts_to_same_blocks(tmp_path) -> None:
    source_dir = tmp_path / "datasets" / "source"
    source_dir.mkdir(parents=True)
    np.arange(27, dtype=np.uint8).tofile(source_dir / "data-00000.bin")
    (source_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dtype": "uint8",
                "shards": [{"bin_file": "data-00000.bin", "token_count": 27}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "mix_manifest.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {"name": "source", "manifest": "datasets/source/manifest.json"}
                ],
                "is_byte_level": True,
            }
        ),
        encoding="utf-8",
    )

    ctx2 = PackedCurriculumByteDataset(
        tmp_path, seq_len=2, base_seq_len=8, shuffle=False
    )
    ctx4 = PackedCurriculumByteDataset(
        tmp_path, seq_len=4, base_seq_len=8, shuffle=False
    )

    assert len(ctx2) == 12
    assert len(ctx4) == 6
    assert [value for index in range(4) for value in ctx2[index]["input_ids"]] == list(
        range(8)
    )
    assert [value for index in range(2) for value in ctx4[index]["input_ids"]] == list(
        range(8)
    )
    assert ctx2[3]["labels"].tolist() == [7, 8]
    assert ctx4[1]["labels"].tolist() == [5, 6, 7, 8]


def test_packed_curriculum_order_is_context_invariant_and_resumable(tmp_path) -> None:
    source_dir = tmp_path / "datasets" / "source"
    source_dir.mkdir(parents=True)
    np.arange(45, dtype=np.uint8).tofile(source_dir / "data-00000.bin")
    (source_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dtype": "uint8",
                "shards": [{"bin_file": "data-00000.bin", "token_count": 45}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "mix_manifest.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {"name": "source", "manifest": "datasets/source/manifest.json"}
                ],
                "is_byte_level": True,
            }
        ),
        encoding="utf-8",
    )

    ctx2 = PackedCurriculumByteDataset(
        tmp_path, seq_len=2, base_seq_len=8, shuffle=True, seed=42
    )
    ctx8 = PackedCurriculumByteDataset(
        tmp_path,
        seq_len=8,
        base_seq_len=8,
        shuffle=True,
        seed=42,
        start_block=2,
    )
    audit2 = ctx2.order_audit()
    audit8 = ctx8.order_audit()

    assert audit2["block_order_sha256"] == audit8["block_order_sha256"]
    assert audit8["start_block"] == 2
    expected_block = audit2["block_prefix"][2]
    resumed_block = audit8["block_prefix"][0]
    assert resumed_block["block_index"] == expected_block["block_index"]
    assert resumed_block["block_offset"] == expected_block["block_offset"]
