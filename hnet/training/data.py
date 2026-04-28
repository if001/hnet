import json
import logging
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


def _resolve_packed_paths(packed_dir: str | Path) -> tuple[Path, Path]:
    base = Path(packed_dir)
    data_path = base / "data.bin"
    metadata_path = base / "metadata.json"
    return data_path, metadata_path


def load_packed_metadata(packed_dir: str | Path) -> dict[str, object]:
    _data_path, metadata_path = _resolve_packed_paths(packed_dir)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found under {packed_dir}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def compute_packed_total_tokens(packed_dir: str | Path) -> int:
    data_path, _metadata_path = _resolve_packed_paths(packed_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"data.bin not found under {packed_dir}")
    return data_path.stat().st_size


class PackedByteDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        packed_dir: str | Path,
        seq_len: int,
        sample_start: int = 0,
        sample_end: int | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.stride = seq_len + 1

        data_path, _metadata_path = _resolve_packed_paths(packed_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"data.bin not found under {packed_dir}")

        self._tokens = np.memmap(data_path, mode="r", dtype=np.uint8)
        total_tokens = int(self._tokens.shape[0])
        total_samples = total_tokens // self.stride

        if sample_end is None:
            sample_end = total_samples
        sample_start = max(0, sample_start)
        sample_end = min(total_samples, sample_end)
        if sample_end < sample_start:
            sample_end = sample_start

        self.sample_start = sample_start
        self.sample_end = sample_end
        self.num_samples = self.sample_end - self.sample_start
        self._mask = torch.ones(self.seq_len, dtype=torch.bool)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)
        sample_index = self.sample_start + index
        start = sample_index * self.stride
        end = start + self.stride
        chunk = np.asarray(self._tokens[start:end], dtype=np.int64)

        input_ids = torch.from_numpy(chunk[:-1].copy()).long()
        labels = torch.from_numpy(chunk[1:].copy()).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "mask": self._mask,
        }
