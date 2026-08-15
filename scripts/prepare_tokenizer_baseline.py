from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers


SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def source_entries(packed_dir: Path) -> list[dict[str, Any]]:
    mix = load_json(packed_dir / "mix_manifest.json")
    datasets = mix.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("mix_manifest datasets must be a list")
    entries = [dict(item) for item in datasets if isinstance(item, Mapping)]
    if not entries:
        raise ValueError("No dataset entries found in byte-packed input")
    return entries


def iter_source_documents(
    packed_dir: Path, entry: Mapping[str, Any]
) -> Iterator[str]:
    manifest_rel = entry.get("manifest")
    if not isinstance(manifest_rel, str):
        raise ValueError("Dataset entry is missing manifest")
    manifest_path = packed_dir / manifest_rel
    manifest = load_json(manifest_path)
    if manifest.get("dtype") != "uint8":
        raise ValueError(f"Expected uint8 byte-packed source: {manifest_path}")
    add_bos = bool(manifest.get("add_bos", True))
    add_eos = bool(manifest.get("add_eos", True))
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError(f"Source shards must be a list: {manifest_path}")

    for shard in shards:
        if not isinstance(shard, Mapping):
            continue
        bin_file = shard.get("bin_file")
        idx_file = shard.get("idx_file")
        if not isinstance(bin_file, str) or not isinstance(idx_file, str):
            raise ValueError(f"Invalid shard entry in {manifest_path}")
        tokens = np.memmap(manifest_path.parent / bin_file, mode="r", dtype=np.uint8)
        offsets = np.fromfile(manifest_path.parent / idx_file, dtype=np.uint64)
        for start, end in zip(offsets[:-1], offsets[1:]):
            raw = bytes(tokens[int(start) : int(end)])
            if add_bos and raw.startswith(bytes([254])):
                raw = raw[1:]
            if add_eos and raw.endswith(bytes([255])):
                raw = raw[:-1]
            if raw:
                yield raw.decode("utf-8")


def iter_all_documents(packed_dir: Path) -> Iterator[str]:
    for entry in source_entries(packed_dir):
        yield from iter_source_documents(packed_dir, entry)


def train_tokenizer(
    packed_dir: Path,
    output_dir: Path,
    vocab_size: int,
    min_frequency: int,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_all_documents(packed_dir), trainer=trainer)
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        raise RuntimeError("Tokenizer training did not create BOS/EOS tokens")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
    )

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    manifest = {
        "format": "hnet_bpe_tokenizer_v1",
        "training_data": str(packed_dir),
        "training_mix_manifest_sha256": sha256_file(
            packed_dir / "mix_manifest.json"
        ),
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "min_frequency": min_frequency,
        "special_tokens": {
            token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS
        },
        "tokenizer_sha256": sha256_file(tokenizer_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return tokenizer_path


def flush_shard(
    source_dir: Path,
    shard_index: int,
    chunks: list[np.ndarray],
    offsets: list[int],
) -> dict[str, Any]:
    stem = f"data-{shard_index:05d}"
    bin_path = source_dir / f"{stem}.bin"
    idx_path = source_dir / f"{stem}.idx"
    with bin_path.open("wb") as handle:
        for chunk in chunks:
            handle.write(chunk.tobytes())
    np.asarray(offsets, dtype=np.uint64).tofile(idx_path)
    return {
        "name": stem,
        "bin_file": bin_path.name,
        "idx_file": idx_path.name,
        "token_count": int(sum(chunk.size for chunk in chunks)),
        "doc_count": len(chunks),
    }


def pack_source(
    byte_packed_dir: Path,
    entry: Mapping[str, Any],
    output_dir: Path,
    tokenizer: Tokenizer,
    max_shard_tokens: int,
) -> dict[str, Any]:
    source_rel = entry.get("source_dir")
    if not isinstance(source_rel, str):
        raise ValueError("Dataset entry is missing source_dir")
    source_dir = output_dir / source_rel
    source_dir.mkdir(parents=True, exist_ok=False)

    shards: list[dict[str, Any]] = []
    chunks: list[np.ndarray] = []
    offsets = [0]
    shard_tokens = 0
    total_tokens = 0
    total_raw_bytes = 0
    total_documents = 0
    shard_index = 0

    for text in iter_source_documents(byte_packed_dir, entry):
        ids = np.asarray(tokenizer.encode(text).ids, dtype=np.uint32)
        if chunks and shard_tokens + ids.size > max_shard_tokens:
            shards.append(flush_shard(source_dir, shard_index, chunks, offsets))
            shard_index += 1
            chunks = []
            offsets = [0]
            shard_tokens = 0
        chunks.append(ids)
        shard_tokens += int(ids.size)
        offsets.append(offsets[-1] + int(ids.size))
        total_tokens += int(ids.size)
        total_raw_bytes += len(text.encode("utf-8"))
        total_documents += 1
    if chunks:
        shards.append(flush_shard(source_dir, shard_index, chunks, offsets))

    source_manifest = {
        "format": "hnet_packed_token_source_v1",
        "dtype": "uint32",
        "tokenizer": "BPE",
        "source": {
            "name": entry.get("name"),
            "config_name": entry.get("config_name"),
            "split": entry.get("split", "train"),
            "take_examples": entry.get("take_examples", -1),
            "skip_examples": entry.get("skip_examples", 0),
        },
        "total_records_seen": total_documents,
        "total_records_used": total_documents,
        "total_tokens": total_tokens,
        "total_raw_bytes": total_raw_bytes,
        "shard_count": len(shards),
        "shards": shards,
    }
    manifest_path = source_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        **dict(entry),
        "source_dir": source_rel,
        "manifest": str(manifest_path.relative_to(output_dir)),
        "total_tokens": total_tokens,
        "total_raw_bytes": total_raw_bytes,
        "shard_count": len(shards),
        "shard_token_counts": [int(shard["token_count"]) for shard in shards],
    }


def build_indices(
    output_dir: Path, shard_token_counts: list[int], seq_len: int, seed: int
) -> dict[str, Any]:
    counts = np.asarray(
        [max(0, count // (seq_len + 1)) for count in shard_token_counts],
        dtype=np.int64,
    )
    shard_ids = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    chunk_offsets = np.concatenate(
        [np.arange(count, dtype=np.int64) for count in counts]
    ) if counts.size else np.empty(0, dtype=np.int64)
    sample_index = np.stack([shard_ids, chunk_offsets], axis=1)
    shuffle_index = np.arange(sample_index.shape[0], dtype=np.int64)
    np.random.default_rng(seed).shuffle(shuffle_index)
    document_index = np.arange(sample_index.shape[0], dtype=np.int64)

    indices_dir = output_dir / "indices" / f"seq_{seq_len}"
    indices_dir.mkdir(parents=True)
    sample_path = indices_dir / "sample_index.npy"
    shuffle_path = indices_dir / f"shuffle_index_seed_{seed}.npy"
    document_path = indices_dir / "document_index.npy"
    np.save(sample_path, sample_index)
    np.save(shuffle_path, shuffle_index)
    np.save(document_path, document_index)
    return {
        "seq_len": seq_len,
        "sample_index": str(sample_path.relative_to(output_dir)),
        "shuffle_index": str(shuffle_path.relative_to(output_dir)),
        "document_index": str(document_path.relative_to(output_dir)),
        "total_samples": int(sample_index.shape[0]),
        "seed": seed,
    }


def token_byte_lengths(tokenizer: Tokenizer) -> list[int]:
    special_ids = {
        tokenizer.token_to_id(token) for token in SPECIAL_TOKENS
    }
    lengths: list[int] = []
    for token_id in range(tokenizer.get_vocab_size()):
        token = tokenizer.id_to_token(token_id)
        lengths.append(0 if token_id in special_ids else len(token or ""))
    return lengths


def pack_tokenized_dataset(
    byte_packed_dir: Path,
    tokenizer_path: Path,
    output_dir: Path,
    max_shard_tokens: int,
    index_seq_len: int,
    seed: int,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    results = [
        pack_source(
            byte_packed_dir,
            entry,
            output_dir,
            tokenizer,
            max_shard_tokens,
        )
        for entry in source_entries(byte_packed_dir)
    ]
    shard_counts = [
        count for result in results for count in result.pop("shard_token_counts")
    ]
    mix = {
        "format": "hnet_packed_token_mix_v1",
        "seed": seed,
        "dtype": "uint32",
        "tokenizer": "BPE",
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "vocab_size": tokenizer.get_vocab_size(),
        "is_byte_level": False,
        "token_byte_lengths": token_byte_lengths(tokenizer),
        "total_records_used": sum(
            int(load_json(output_dir / result["manifest"])["total_records_used"])
            for result in results
        ),
        "total_tokens": sum(int(result["total_tokens"]) for result in results),
        "total_raw_bytes": sum(int(result["total_raw_bytes"]) for result in results),
        "total_shards": sum(int(result["shard_count"]) for result in results),
        "datasets": results,
        "index": build_indices(output_dir, shard_counts, index_seq_len, seed),
    }
    (output_dir / "mix_manifest.json").write_text(
        json.dumps(mix, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and pack the parameter-matched tokenizer baseline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--byte-packed-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--vocab-size", type=int, default=128_000)
    train_parser.add_argument("--min-frequency", type=int, default=2)

    pack_parser = subparsers.add_parser("pack")
    pack_parser.add_argument("--byte-packed-dir", type=Path, required=True)
    pack_parser.add_argument("--tokenizer-path", type=Path, required=True)
    pack_parser.add_argument("--output-dir", type=Path, required=True)
    pack_parser.add_argument("--max-shard-tokens", type=int, default=50_000_000)
    pack_parser.add_argument("--index-seq-len", type=int, default=2048)
    pack_parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_tokenizer(
            args.byte_packed_dir,
            args.output_dir,
            args.vocab_size,
            args.min_frequency,
        )
    else:
        pack_tokenized_dataset(
            args.byte_packed_dir,
            args.tokenizer_path,
            args.output_dir,
            args.max_shard_tokens,
            args.index_seq_len,
            args.seed,
        )


if __name__ == "__main__":
    main()
