from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from hnet.utils.tokenizers import ByteTokenizer

from .dataset import SFTDataConfig, build_sft_train_dataset


class StreamingSFTByteDataset(IterableDataset):
    def __init__(
        self,
        *,
        seq_len: int,
        shuffle_buffer_size: int,
        seed: int,
        chat_tokenizer_path: str,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.byte_tokenizer = ByteTokenizer()
        self.chat_tokenizer_path = chat_tokenizer_path
        self._chat_tokenizer: PreTrainedTokenizerBase | None = None

    def _get_chat_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._chat_tokenizer is None:
            self._chat_tokenizer = AutoTokenizer.from_pretrained(
                self.chat_tokenizer_path
            )
        return self._chat_tokenizer

    def _iter_texts(self) -> Iterator[str]:
        sample_cfg = SFTDataConfig(
            seed=self.seed,
            shuffle_buffer_size=self.shuffle_buffer_size,
        )
        dataset = build_sft_train_dataset(sample_cfg)
        tokenizer = self._get_chat_tokenizer()

        for record in dataset:
            messages = record.get("messages")
            if not isinstance(messages, list):
                continue
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if text:
                yield text

    def __iter__(self):
        token_buffer: list[int] = []

        for text in self._iter_texts():
            encoded = self.byte_tokenizer.encode([text], add_bos=True, add_eos=True)[0][
                "input_ids"
            ].tolist()
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
