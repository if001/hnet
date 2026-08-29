import hashlib
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
    dtype: str = "uint8"


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
        dtype = str(source_manifest.get("dtype", "uint8"))
        if dtype not in {"uint8", "uint16", "uint32"}:
            raise ValueError(f"Unsupported packed token dtype: {dtype}")
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
                    dtype=dtype,
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

        mix = load_mix_manifest(self.packed_dir)
        raw_byte_lengths = mix.get("token_byte_lengths")
        if isinstance(raw_byte_lengths, list):
            self._token_byte_lengths = np.asarray(raw_byte_lengths, dtype=np.int64)
            self.is_byte_level = bool(mix.get("is_byte_level", False))
        else:
            self._token_byte_lengths = np.ones(256, dtype=np.int64)
            self.is_byte_level = True

        all_shards = list_packed_shards(self.packed_dir)
        if shard_indices is not None:
            self.shards = [
                all_shards[i] for i in shard_indices if 0 <= int(i) < len(all_shards)
            ]
            self._has_shard_subset = True
        else:
            self.shards = all_shards
            self._has_shard_subset = False
        dtypes = {shard.dtype for shard in self.shards}
        if len(dtypes) > 1:
            raise ValueError(f"Packed shards use mixed token dtypes: {sorted(dtypes)}")
        self._chunk_counts = np.asarray(
            [int(max(0, shard.token_count // self.stride)) for shard in self.shards],
            dtype=np.int64,
        )
        self.total_chunks = int(self._chunk_counts.sum())
        self._shard_memmaps: list[np.memmap] = [
            np.memmap(shard.bin_path, mode="r", dtype=np.dtype(shard.dtype))
            for shard in self.shards
        ]
        self._sample_shard_ids, self._sample_chunk_offsets, self._shuffle_index = (
            self._build_or_load_indices()
        )
        self.total_chunks = int(self._shuffle_index.shape[0])

        self._start = min(self.start_micro_batch, self.total_chunks)

    def __len__(self) -> int:
        return self.total_chunks - self._start

    def order_audit(self, sample_limit: int = 32) -> dict[str, object]:
        """Return a stable digest and an inspectable prefix of packed sample order."""
        canonical = np.asarray(self._shuffle_index, dtype="<i8")
        digest = hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
        samples: list[dict[str, int]] = []
        stop = min(self._start + max(0, sample_limit), self.total_chunks)
        for position in range(self._start, stop):
            sample_index = int(self._shuffle_index[position])
            samples.append(
                {
                    "position": position,
                    "sample_index": sample_index,
                    "shard_id": int(self._sample_shard_ids[sample_index]),
                    "chunk_offset": int(self._sample_chunk_offsets[sample_index]),
                }
            )
        return {
            "seed": self.seed,
            "shuffle": self.shuffle,
            "start_micro_batch": self._start,
            "total_chunks": self.total_chunks,
            "shuffle_index_sha256": digest,
            "sample_prefix": samples,
        }

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
        targets = chunk[1:]
        if targets.size and int(targets.max()) >= self._token_byte_lengths.shape[0]:
            raise ValueError(
                "Packed token id exceeds token_byte_lengths lookup: "
                f"max_id={int(targets.max())} lookup={self._token_byte_lengths.shape[0]}"
            )

        input_ids = torch.from_numpy(chunk[:-1].copy()).long()
        labels = torch.from_numpy(targets.copy()).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "mask": self._mask,
            "target_byte_lengths": torch.from_numpy(
                self._token_byte_lengths[targets].copy()
            ).long(),
            "is_byte_level": torch.tensor(self.is_byte_level, dtype=torch.bool),
        }


class PackedCurriculumByteDataset(torch.utils.data.Dataset):
    """Read aligned sub-sequences from a shuffled canonical long-context stream.

    A canonical block contains ``base_seq_len + 1`` tokens. Shorter contexts
    split that block into adjacent, one-token-overlapping training examples.
    When the DataLoader batch size is ``base_seq_len // seq_len``, every batch
    consumes exactly one canonical block at every curriculum context length.
    """

    def __init__(
        self,
        packed_dir: str | Path,
        seq_len: int,
        base_seq_len: int,
        shuffle: bool = True,
        seed: int = 42,
        shard_indices: Sequence[int] | None = None,
        start_block: int = 0,
    ) -> None:
        super().__init__()
        if seq_len <= 0 or base_seq_len <= 0:
            raise ValueError("seq_len and base_seq_len must be positive")
        if base_seq_len % seq_len != 0:
            raise ValueError(
                f"base_seq_len={base_seq_len} must be divisible by seq_len={seq_len}"
            )

        self.packed_dir = Path(packed_dir)
        self.seq_len = int(seq_len)
        self.base_seq_len = int(base_seq_len)
        self.subsequences_per_block = self.base_seq_len // self.seq_len
        self.base_stride = self.base_seq_len + 1
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._mask = torch.ones(self.seq_len, dtype=torch.bool)

        mix = load_mix_manifest(self.packed_dir)
        raw_byte_lengths = mix.get("token_byte_lengths")
        if isinstance(raw_byte_lengths, list):
            self._token_byte_lengths = np.asarray(raw_byte_lengths, dtype=np.int64)
            self.is_byte_level = bool(mix.get("is_byte_level", False))
        else:
            self._token_byte_lengths = np.ones(256, dtype=np.int64)
            self.is_byte_level = True

        all_shards = list_packed_shards(self.packed_dir)
        if shard_indices is None:
            self.shards = all_shards
        else:
            self.shards = [
                all_shards[index]
                for index in shard_indices
                if 0 <= int(index) < len(all_shards)
            ]
        dtypes = {shard.dtype for shard in self.shards}
        if len(dtypes) > 1:
            raise ValueError(f"Packed shards use mixed token dtypes: {sorted(dtypes)}")

        self._block_counts = np.asarray(
            [max(0, shard.token_count // self.base_stride) for shard in self.shards],
            dtype=np.int64,
        )
        self.total_blocks = int(self._block_counts.sum())
        self._block_shard_ids = np.repeat(
            np.arange(len(self.shards), dtype=np.int64), self._block_counts
        )
        self._block_offsets = np.empty(self.total_blocks, dtype=np.int64)
        cursor = 0
        for count in self._block_counts.tolist():
            self._block_offsets[cursor : cursor + count] = np.arange(
                count, dtype=np.int64
            )
            cursor += count
        self._block_order = np.arange(self.total_blocks, dtype=np.int64)
        if self.shuffle and self.total_blocks > 1:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self._block_order)

        self.start_block = min(max(0, int(start_block)), self.total_blocks)
        self._shard_memmaps: list[np.memmap] = [
            np.memmap(shard.bin_path, mode="r", dtype=np.dtype(shard.dtype))
            for shard in self.shards
        ]

    def __len__(self) -> int:
        return (
            self.total_blocks - self.start_block
        ) * self.subsequences_per_block

    def order_audit(self, sample_limit: int = 32) -> dict[str, object]:
        canonical = np.asarray(self._block_order, dtype="<i8")
        digest = hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
        blocks: list[dict[str, int]] = []
        stop = min(
            self.start_block + max(0, sample_limit),
            self.total_blocks,
        )
        for position in range(self.start_block, stop):
            block_index = int(self._block_order[position])
            blocks.append(
                {
                    "position": position,
                    "block_index": block_index,
                    "shard_id": int(self._block_shard_ids[block_index]),
                    "block_offset": int(self._block_offsets[block_index]),
                }
            )
        return {
            "mode": "aligned_context_curriculum",
            "seed": self.seed,
            "shuffle": self.shuffle,
            "seq_len": self.seq_len,
            "base_seq_len": self.base_seq_len,
            "subsequences_per_block": self.subsequences_per_block,
            "start_block": self.start_block,
            "total_blocks": self.total_blocks,
            "block_order_sha256": digest,
            "block_prefix": blocks,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        relative_block = index // self.subsequences_per_block
        subsequence = index % self.subsequences_per_block
        order_position = self.start_block + relative_block
        block_index = int(self._block_order[order_position])
        shard_index = int(self._block_shard_ids[block_index])
        block_offset = int(self._block_offsets[block_index])
        start = block_offset * self.base_stride + subsequence * self.seq_len
        end = start + self.seq_len + 1
        chunk = np.asarray(
            self._shard_memmaps[shard_index][start:end], dtype=np.int64
        )
        if chunk.size != self.seq_len + 1:
            raise RuntimeError(
                f"incomplete curriculum chunk: expected={self.seq_len + 1} "
                f"actual={chunk.size}"
            )
        targets = chunk[1:]
        if targets.size and int(targets.max()) >= self._token_byte_lengths.shape[0]:
            raise ValueError(
                "Packed token id exceeds token_byte_lengths lookup: "
                f"max_id={int(targets.max())} "
                f"lookup={self._token_byte_lengths.shape[0]}"
            )

        return {
            "input_ids": torch.from_numpy(chunk[:-1].copy()).long(),
            "labels": torch.from_numpy(targets.copy()).long(),
            "mask": self._mask,
            "target_byte_lengths": torch.from_numpy(
                self._token_byte_lengths[targets].copy()
            ).long(),
            "is_byte_level": torch.tensor(self.is_byte_level, dtype=torch.bool),
            "curriculum_block_position": torch.tensor(
                order_position, dtype=torch.long
            ),
            "curriculum_subsequence": torch.tensor(subsequence, dtype=torch.long),
        }
