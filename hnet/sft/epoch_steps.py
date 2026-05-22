from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch
from torch.utils.data import DataLoader

from .data import StreamingSFTByteDataset


@dataclass(frozen=True)
class SFTEpochEstimate:
    micro_batches: int
    samples: int
    optimizer_steps: int


def estimate_sft_epoch_steps(
    *,
    seq_len: int,
    packing: bool,
    batch_size: int,
    grad_accum_steps: int,
    num_workers: int,
    shuffle_buffer_size: int,
    seed: int,
    chat_tokenizer_path: str,
    mix_config_path: str | None,
) -> SFTEpochEstimate:
    dataset = StreamingSFTByteDataset(
        seq_len=seq_len,
        packing=packing,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        chat_tokenizer_path=chat_tokenizer_path,
        mix_config_path=mix_config_path,
    )
    dataloader_kwargs: dict[str, object] = {}
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **dataloader_kwargs,
    )

    micro_batches = 0
    samples = 0
    for micro_batches, batch in enumerate(dataloader, start=1):
        samples += int(batch["input_ids"].shape[0])

    optimizer_steps = ceil(micro_batches / grad_accum_steps) if micro_batches > 0 else 0
    return SFTEpochEstimate(
        micro_batches=micro_batches,
        samples=samples,
        optimizer_steps=optimizer_steps,
    )
