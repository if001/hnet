from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import IterableDataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from hnet.utils.tokenizers import ByteTokenizer

from .dataset import SFTDataConfig, build_sft_train_dataset


class StreamingSFTByteDataset(IterableDataset):
    def __init__(
        self,
        *,
        seq_len: int,
        packing: bool,
        shuffle_buffer_size: int,
        seed: int,
        chat_tokenizer_path: str,
        model_tokenizer_path: str | None = None,
        mix_config_path: str | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.packing = packing
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.byte_tokenizer = ByteTokenizer()
        self.chat_tokenizer_path = chat_tokenizer_path
        self.model_tokenizer_path = model_tokenizer_path
        self.mix_config_path = mix_config_path
        self._chat_tokenizer: PreTrainedTokenizerBase | None = None
        self._model_tokenizer: Tokenizer | None = None

    def _get_chat_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._chat_tokenizer is None:
            self._chat_tokenizer = AutoTokenizer.from_pretrained(
                self.chat_tokenizer_path
            )
        return self._chat_tokenizer

    def _get_model_tokenizer(self) -> Tokenizer | None:
        if self.model_tokenizer_path is None:
            return None
        if self._model_tokenizer is None:
            self._model_tokenizer = Tokenizer.from_file(self.model_tokenizer_path)
        return self._model_tokenizer

    def _encode(self, text: str) -> list[int]:
        tokenizer = self._get_model_tokenizer()
        if tokenizer is not None:
            return tokenizer.encode(text).ids
        return self.byte_tokenizer.encode(
            [text], add_bos=True, add_eos=True
        )[0]["input_ids"].tolist()

    def _eos_id(self) -> int:
        tokenizer = self._get_model_tokenizer()
        if tokenizer is not None:
            eos_id = tokenizer.token_to_id("<eos>")
            if eos_id is None:
                raise ValueError("Model tokenizer is missing <eos>")
            return eos_id
        return int(self.byte_tokenizer.eos_idx)

    def _iter_texts(self) -> Iterator[str]:
        sample_cfg = SFTDataConfig(
            seed=self.seed,
            shuffle_buffer_size=self.shuffle_buffer_size,
            mix_config_path=self.mix_config_path,
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
        if self.packing:
            token_buffer: list[int] = []
            for text in self._iter_texts():
                encoded = self._encode(text)
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
            return

        eos_id = self._eos_id()
        max_tokens = self.seq_len + 1
        for text in self._iter_texts():
            encoded = self._encode(text)
            if len(encoded) < 2:
                continue

            encoded = encoded[:max_tokens]
            real_len = len(encoded) - 1
            if real_len <= 0:
                continue

            if len(encoded) < max_tokens:
                encoded = encoded + [eos_id] * (max_tokens - len(encoded))

            input_ids = torch.tensor(encoded[:-1], dtype=torch.long)
            labels = torch.full((self.seq_len,), -100, dtype=torch.long)
            labels[:real_len] = torch.tensor(encoded[1 : 1 + real_len], dtype=torch.long)
            mask = torch.zeros(self.seq_len, dtype=torch.bool)
            mask[:real_len] = True
            yield {
                "input_ids": input_ids,
                "labels": labels,
                "mask": mask,
            }
