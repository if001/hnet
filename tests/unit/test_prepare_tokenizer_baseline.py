import json

import numpy as np

from hnet.training.data import PackedMixByteDataset
from scripts.prepare_tokenizer_baseline import (
    iter_all_documents,
    pack_tokenized_dataset,
    train_tokenizer,
)


def _write_byte_packed_fixture(root) -> None:
    source = root / "datasets" / "source"
    source.mkdir(parents=True)
    documents = ["abc abc", "日本語の文"]
    chunks = [bytes([254]) + text.encode("utf-8") + bytes([255]) for text in documents]
    offsets = [0]
    with (source / "data-00000.bin").open("wb") as handle:
        for chunk in chunks:
            handle.write(chunk)
            offsets.append(offsets[-1] + len(chunk))
    np.asarray(offsets, dtype=np.uint64).tofile(source / "data-00000.idx")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "dtype": "uint8",
                "add_bos": True,
                "add_eos": True,
                "shards": [
                    {
                        "bin_file": "data-00000.bin",
                        "idx_file": "data-00000.idx",
                        "token_count": offsets[-1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "mix_manifest.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "fixture",
                        "source_dir": "datasets/source",
                        "manifest": "datasets/source/manifest.json",
                        "weight": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_train_and_pack_tokenizer_baseline_from_byte_snapshot(tmp_path) -> None:
    byte_root = tmp_path / "byte"
    tokenizer_root = tmp_path / "tokenizer"
    token_root = tmp_path / "tokens"
    _write_byte_packed_fixture(byte_root)

    assert list(iter_all_documents(byte_root)) == ["abc abc", "日本語の文"]
    tokenizer_path = train_tokenizer(
        byte_root,
        tokenizer_root,
        vocab_size=64,
        min_frequency=1,
    )
    pack_tokenized_dataset(
        byte_root,
        tokenizer_path,
        token_root,
        max_shard_tokens=100,
        index_seq_len=3,
        seed=42,
    )

    manifest = json.loads(
        (token_root / "mix_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dtype"] == "uint32"
    assert manifest["total_raw_bytes"] == len("abc abc日本語の文".encode("utf-8"))
    assert len(manifest["token_byte_lengths"]) == manifest["vocab_size"]

    dataset = PackedMixByteDataset(token_root, seq_len=3, shuffle=False)
    assert len(dataset) > 0
    assert not bool(dataset[0]["is_byte_level"])
