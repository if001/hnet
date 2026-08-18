import json

import numpy as np

from hnet.training.data import DefaultRecordFormatter, PackedMixByteDataset


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
