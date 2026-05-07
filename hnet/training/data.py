import json
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import IterableDataset as HFIterableDataset
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import get_worker_info

from ..utils.tokenizers import ByteTokenizer
from .config import DatasetSource
from .record_formatter import DefaultRecordFormatter, RecordFormatter
logger = logging.getLogger(__name__)


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


class PackedMixByteDataset(torch.utils.data.Dataset):
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
            self._has_shard_subset = True
        else:
            self.shards = all_shards
            self._has_shard_subset = False
        self._chunk_counts = np.asarray(
            [int(max(0, shard.token_count // self.stride)) for shard in self.shards],
            dtype=np.int64,
        )
        self.total_chunks = int(self._chunk_counts.sum())
        self._shard_memmaps: list[np.memmap] = [
            np.memmap(shard.bin_path, mode="r", dtype=np.uint8) for shard in self.shards
        ]
        self._sample_shard_ids, self._sample_chunk_offsets, self._shuffle_index = (
            self._build_or_load_indices()
        )
        self.total_chunks = int(self._shuffle_index.shape[0])

        self._start = min(self.start_micro_batch, self.total_chunks)

    def __len__(self) -> int:
        return self.total_chunks - self._start

    def _build_or_load_indices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mix = load_mix_manifest(self.packed_dir)
        if self._has_shard_subset:
            index_info = None
        else:
            index_info = mix.get("index")
        if isinstance(index_info, Mapping):
            seq_len = index_info.get("seq_len")
            seed = index_info.get("seed")
            sample_index_rel = index_info.get("sample_index")
            shuffle_index_rel = index_info.get("shuffle_index")
            if (
                isinstance(seq_len, int)
                and isinstance(seed, int)
                and isinstance(sample_index_rel, str)
                and isinstance(shuffle_index_rel, str)
                and seq_len == self.seq_len
                and seed == self.seed
            ):
                sample_index_path = self.packed_dir / sample_index_rel
                shuffle_index_path = self.packed_dir / shuffle_index_rel
                if sample_index_path.exists() and shuffle_index_path.exists():
                    sample_index = np.load(sample_index_path, mmap_mode="r")
                    shuffle_index = np.load(shuffle_index_path, mmap_mode="r")
                    sample_shard_ids = np.asarray(sample_index[:, 0], dtype=np.int64)
                    sample_chunk_offsets = np.asarray(sample_index[:, 1], dtype=np.int64)
                    shuffle_array = np.asarray(shuffle_index, dtype=np.int64)
                    if not self.shuffle:
                        shuffle_array = np.arange(sample_shard_ids.shape[0], dtype=np.int64)
                    return sample_shard_ids, sample_chunk_offsets, shuffle_array

        # Fallback: build indices at runtime.
        shard_ids = np.repeat(
            np.arange(len(self.shards), dtype=np.int64),
            self._chunk_counts,
        )
        sample_chunk_offsets = np.empty(self.total_chunks, dtype=np.int64)
        cursor = 0
        for count in self._chunk_counts.tolist():
            sample_chunk_offsets[cursor : cursor + count] = np.arange(
                count, dtype=np.int64
            )
            cursor += count
        shuffle_index = np.arange(self.total_chunks, dtype=np.int64)
        if self.shuffle and self.total_chunks > 1:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(shuffle_index)
        return shard_ids, sample_chunk_offsets, shuffle_index

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        shuffled_sample_index = int(self._shuffle_index[self._start + index])
        shard_idx = int(self._sample_shard_ids[shuffled_sample_index])
        chunk_idx = int(self._sample_chunk_offsets[shuffled_sample_index])
        start = chunk_idx * self.stride
        end = start + self.stride
        mm = self._shard_memmaps[shard_idx]
        chunk = np.asarray(mm[start:end], dtype=np.int64)

        input_ids = torch.from_numpy(chunk[:-1].copy()).long()
        labels = torch.from_numpy(chunk[1:].copy()).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "mask": self._mask,
        }
