import json
import logging
import random
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from datasets import IterableDataset as HFIterableDataset
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import get_worker_info

from ..utils.tokenizers import ByteTokenizer
from .config import DatasetSource

TextValue = str | int | float | bool | None

PREFERRED_TEXT_KEYS = (
    "text",
    "content",
    "completion",
    "response",
    "output",
    "answer",
    "prompt",
    "instruction",
    "question",
    "input",
    "context",
    "title",
)
logger = logging.getLogger(__name__)


class RecordFormatter(Protocol):
    def format_record(self, record: Mapping[str, object]) -> str | None: ...


def _stringify_scalar(value: TextValue) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _join_message_content(messages: Sequence[object]) -> str:
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = _stringify_scalar(message.get("role"))
        content = _stringify_value(message.get("content"))
        text = content.strip()
        if not text:
            continue
        if role:
            parts.append(f"{role}: {text}")
        else:
            parts.append(text)
    return "\n".join(parts)


def _stringify_mapping(value: Mapping[str, object]) -> str:
    messages = value.get("messages")
    if isinstance(messages, Sequence) and not isinstance(
        messages, (str, bytes, bytearray)
    ):
        joined = _join_message_content(messages)
        if joined:
            return joined
    if "content" in value:
        return _stringify_value(value["content"])
    if "text" in value:
        return _stringify_value(value["text"])
    try:
        return json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
    except TypeError:
        return ""


def _stringify_sequence(value: Sequence[object]) -> str:
    if not value:
        return ""
    if all(isinstance(item, Mapping) for item in value):
        return _join_message_content(value)
    parts = [_stringify_value(item).strip() for item in value]
    return "\n".join(part for part in parts if part)


def _stringify_value(value: object) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _stringify_scalar(value)
    if isinstance(value, Mapping):
        return _stringify_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _stringify_sequence(value)
    return ""


@dataclass(frozen=True)
class DefaultRecordFormatter:
    preferred_keys: tuple[str, ...] = PREFERRED_TEXT_KEYS

    def format_record(self, record: Mapping[str, object]) -> str | None:
        ordered_values: list[str] = []

        for key in self.preferred_keys:
            if key not in record:
                continue
            value = _stringify_value(record[key]).strip()
            if value:
                ordered_values.append(value)

        if not ordered_values:
            for key, value in record.items():
                if key.startswith("_"):
                    continue
                rendered = _stringify_value(value).strip()
                if rendered:
                    ordered_values.append(rendered)

        if not ordered_values:
            return None

        deduped_values = list(dict.fromkeys(ordered_values))
        return "\n\n".join(deduped_values)


def _load_streaming_source(
    source: DatasetSource,
) -> HFIterableDataset:
    dataset = load_dataset(
        source.name,
        source.config_name,
        split=source.split,
        streaming=True,
    )

    if source.skip_examples > 0:
        dataset = dataset.skip(source.skip_examples)

    if source.take_examples > 0:
        dataset = dataset.take(source.take_examples)
    return dataset


class StreamingByteDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        sources: Sequence[DatasetSource],
        seq_len: int,
        formatter: RecordFormatter,
        shuffle_buffer_size: int = 512,
        add_bos: bool = True,
        add_eos: bool = True,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.sources = list(sources)
        self.seq_len = seq_len
        self.formatter = formatter
        self.shuffle_buffer_size = shuffle_buffer_size
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.tokenizer = ByteTokenizer()
        self.shuffle = shuffle

    def _iter_stream(self) -> Iterable[Mapping[str, object]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        datasets = [
            _load_streaming_source(source)
            for source in self.sources
        ]
        if len(datasets) == 1:
            merged = datasets[0]
        else:
            merged = concatenate_datasets(datasets)

        if num_workers > 1 and hasattr(merged, "shard"):
            try:
                merged = merged.shard(num_shards=num_workers, index=worker_id)
            except Exception as exc:  # pragma: no cover - runtime backend dependent
                logger.warning(
                    "Failed to shard iterable dataset for worker=%d/%d: %s. "
                    "Falling back to unsharded stream.",
                    worker_id,
                    num_workers,
                    exc,
                )

        if self.shuffle:
            return merged.shuffle(buffer_size=self.shuffle_buffer_size, seed=42)
        return merged

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        token_buffer: list[int] = []
        for record in self._iter_stream():
            text = self.formatter.format_record(record)
            if not text:
                continue

            encoded = self.tokenizer.encode(
                [text],
                add_bos=self.add_bos,
                add_eos=self.add_eos,
            )[0]["input_ids"].tolist()
            token_buffer.extend(encoded)

            while len(token_buffer) >= self.seq_len + 1:
                chunk = token_buffer[: self.seq_len + 1]
                del token_buffer[: self.seq_len + 1]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                mask = torch.ones(self.seq_len, dtype=torch.bool)
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "mask": mask,
                }


def load_mix_manifest(packed_dir: str | Path) -> dict[str, object]:
    base = Path(packed_dir)
    mix_path = base / "mix_manifest.json"
    if not mix_path.exists():
        raise FileNotFoundError(f"mix_manifest.json not found under {packed_dir}")
    return json.loads(mix_path.read_text(encoding="utf-8"))


def compute_packed_total_tokens(packed_dir: str | Path) -> int:
    base = Path(packed_dir)
    mix = load_mix_manifest(base)
    return int(mix.get("total_tokens", 0))


@dataclass(frozen=True)
class PackedShard:
    dataset_name: str
    bin_path: Path
    token_count: int


def _load_shards_from_mix_manifest(packed_dir: str | Path) -> list[PackedShard]:
    base = Path(packed_dir)
    mix = load_mix_manifest(base)
    datasets = mix.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("Invalid mix_manifest.json: datasets must be a list")

    shards: list[PackedShard] = []
    for dataset_entry in datasets:
        if not isinstance(dataset_entry, Mapping):
            continue
        dataset_name = str(dataset_entry.get("name", "dataset"))
        manifest_rel = dataset_entry.get("manifest")
        if not isinstance(manifest_rel, str):
            continue
        manifest_path = base / manifest_rel
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_dir = manifest_path.parent
        source_shards = source_manifest.get("shards", [])
        if not isinstance(source_shards, list):
            continue
        for shard in source_shards:
            if not isinstance(shard, Mapping):
                continue
            bin_file = shard.get("bin_file")
            token_count = shard.get("token_count")
            if not isinstance(bin_file, str) or not isinstance(token_count, int):
                continue
            bin_path = source_dir / bin_file
            shards.append(
                PackedShard(
                    dataset_name=dataset_name,
                    bin_path=bin_path,
                    token_count=int(token_count),
                )
            )
    return shards


def list_packed_shards(packed_dir: str | Path) -> list[PackedShard]:
    return _load_shards_from_mix_manifest(Path(packed_dir))


class PackedMixByteDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        packed_dir: str | Path,
        seq_len: int,
        shuffle: bool = True,
        seed: int = 42,
        shard_indices: Sequence[int] | None = None,
        start_micro_batch: int = 0,
    ) -> None:
        super().__init__()
        self.packed_dir = Path(packed_dir)
        self.seq_len = seq_len
        self.stride = seq_len + 1
        self.shuffle = shuffle
        self.seed = seed
        self.start_micro_batch = max(0, int(start_micro_batch))
        self._mask = torch.ones(self.seq_len, dtype=torch.bool)

        all_shards = list_packed_shards(self.packed_dir)
        if shard_indices is not None:
            self.shards = [
                all_shards[i] for i in shard_indices if 0 <= int(i) < len(all_shards)
            ]
        else:
            self.shards = all_shards
        self.total_chunks = sum(
            int(max(0, shard.token_count // self.stride)) for shard in self.shards
        )

    def __len__(self) -> int:
        return self.total_chunks

    def _iter_worker_shards(self) -> list[PackedShard]:
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        if num_workers <= 1:
            selected = list(self.shards)
        else:
            selected = [
                shard for idx, shard in enumerate(self.shards) if idx % num_workers == worker_id
            ]
        if self.shuffle:
            rng = random.Random(self.seed + worker_id)
            rng.shuffle(selected)
        return selected

    def _compute_worker_shard_layout(
        self, shards: Sequence[PackedShard]
    ) -> tuple[list[int], list[int]]:
        shard_chunk_counts = [
            int(max(0, shard.token_count // self.stride)) for shard in shards
        ]
        prefix = [0]
        for count in shard_chunk_counts:
            prefix.append(prefix[-1] + count)
        return shard_chunk_counts, prefix

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        shards = self._iter_worker_shards()
        shard_chunk_counts, prefix = self._compute_worker_shard_layout(shards)
        worker_total_chunks = prefix[-1] if prefix else 0
        if worker_total_chunks <= 0:
            return

        worker_start = min(self.start_micro_batch, worker_total_chunks)
        start_shard_idx = max(0, bisect_right(prefix, worker_start) - 1)
        start_offset = worker_start - prefix[start_shard_idx]

        for local_idx, shard in enumerate(shards):
            if local_idx < start_shard_idx:
                continue
            if not shard.bin_path.exists():
                continue
            mm = np.memmap(shard.bin_path, mode="r", dtype=np.uint8)
            chunk_count = shard_chunk_counts[local_idx]
            local_start = start_offset if local_idx == start_shard_idx else 0
            for chunk_idx in range(local_start, chunk_count):
                start = chunk_idx * self.stride
                end = start + self.stride
                chunk = np.asarray(mm[start:end], dtype=np.int64)
                input_ids = torch.from_numpy(chunk[:-1].copy()).long()
                labels = torch.from_numpy(chunk[1:].copy()).long()
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "mask": self._mask,
                }
