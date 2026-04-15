from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterator, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import interleave_datasets, load_dataset
from datasets.iterable_dataset import IterableDataset as HFIterableDataset
from transformers import PreTrainedTokenizerBase


ChatMessage = dict[str, str]


@dataclass(frozen=True)
class SFTDataConfig:
    shuffle_buffer_size: int = 10_000
    batch_size: int = 2
    max_length: int = 2048
    num_workers: int = 0
    seed: int = 42

    # Qwen3-style system prompt
    system_prompt: str = "You are a helpful assistant."

    # example数ベースの概算比率
    # JA : EN : CODE = 8 : 1 : 1
    # JA = magpie + JaMARD
    magpie_take: int = 60_000
    jamard_take: int = 35_000
    oasst2_take: int = 15_000
    aya_en_take: int = 15_000
    coding_take: int = 15_000


def _is_mapping_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(v, Mapping) for v in value)


def _normalize_role(role: str) -> str:
    role_lower = role.strip().lower()
    mapping = {
        "human": "user",
        "user": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "system": "system",
    }
    if role_lower not in mapping:
        raise ValueError(f"Unsupported role: {role}")
    return mapping[role_lower]


def _normalize_messages(raw_messages: object) -> list[ChatMessage]:
    if not _is_mapping_list(raw_messages):
        raise ValueError("raw_messages must be a list of mappings")

    messages: list[ChatMessage] = []
    for item in raw_messages:
        if "role" in item and "content" in item:
            role = _normalize_role(str(item["role"]))
            content = str(item["content"]).strip()
        elif "from" in item and "value" in item:
            role = _normalize_role(str(item["from"]))
            content = str(item["value"]).strip()
        else:
            raise ValueError(f"Unsupported message schema: {item}")

        if content:
            messages.append({"role": role, "content": content})

    return messages


def _is_valid_dialogue(messages: Sequence[ChatMessage]) -> bool:
    if len(messages) < 2:
        return False
    if messages[-1]["role"] != "assistant":
        return False
    return any(message["role"] == "user" for message in messages)


def _prepend_qwen3_system(
    messages: Sequence[ChatMessage],
    think_mode: bool,
    system_prompt: str,
) -> list[ChatMessage]:
    control = "/think" if think_mode else "/no_think"

    if messages and messages[0]["role"] == "system":
        merged_system = messages[0]["content"].strip()
        merged_system = f"{merged_system}\n{control}" if merged_system else control
        return [{"role": "system", "content": merged_system}, *messages[1:]]

    system_content = f"{system_prompt}\n{control}".strip() if system_prompt else control
    return [{"role": "system", "content": system_content}, *messages]


_ANSWER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\n(?:最終)?(?:回答|答え)\s*[:：]\s*(.+)$", re.DOTALL),
    re.compile(r"\n####\s*(.+)$", re.DOTALL),
    re.compile(r"\nFinal\s+Answer\s*[:：]\s*(.+)$", re.DOTALL | re.IGNORECASE),
)


def _split_reasoning_and_answer(text: str) -> tuple[str | None, str]:
    cleaned = text.strip()
    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            thought = cleaned[: match.start()].strip()
            answer = match.group(1).strip()
            if thought and answer:
                return thought, answer
    return None, cleaned


def _format_reasoning_assistant(text: str) -> str:
    cleaned = text.strip()
    if "<think>" in cleaned and "</think>" in cleaned:
        return cleaned

    thought, answer = _split_reasoning_and_answer(cleaned)
    if thought is None:
        return cleaned

    return f"<think>\n{thought}\n</think>\n\n{answer}"


def _map_magpie(example: Mapping[str, object], system_prompt: str) -> dict[str, object]:
    messages = _normalize_messages(example["conversations"])
    messages = _prepend_qwen3_system(
        messages, think_mode=False, system_prompt=system_prompt
    )
    return {"messages": messages}


def _map_oasst2(example: Mapping[str, object], system_prompt: str) -> dict[str, object]:
    if "conversations" in example:
        messages = _normalize_messages(example["conversations"])
    elif "messages" in example:
        messages = _normalize_messages(example["messages"])
    else:
        raise KeyError(
            f"Unsupported oasst2 schema. Available keys: {sorted(example.keys())}"
        )

    messages = _prepend_qwen3_system(
        messages,
        think_mode=False,
        system_prompt=system_prompt,
    )
    return {"messages": messages}


def _map_jamard(example: Mapping[str, object], system_prompt: str) -> dict[str, object]:
    messages = _normalize_messages(example["messages"])

    out: list[ChatMessage] = []
    for message in messages:
        if message["role"] == "assistant":
            out.append(
                {
                    "role": "assistant",
                    "content": _format_reasoning_assistant(message["content"]),
                }
            )
        else:
            out.append(message)

    out = _prepend_qwen3_system(out, think_mode=True, system_prompt=system_prompt)
    return {"messages": out}


def _map_aya_english(
    example: Mapping[str, object], system_prompt: str
) -> dict[str, object]:
    user_text = str(example["inputs"]).strip()
    assistant_text = str(example["targets"]).strip()

    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    messages = _prepend_qwen3_system(
        messages, think_mode=False, system_prompt=system_prompt
    )
    return {"messages": messages}


def _map_coding(example: Mapping[str, object], system_prompt: str) -> dict[str, object]:
    messages = _normalize_messages(example["messages"])
    messages = _prepend_qwen3_system(
        messages, think_mode=False, system_prompt=system_prompt
    )
    return {"messages": messages}


def _valid_example(example: Mapping[str, object]) -> bool:
    messages = example.get("messages")
    if not _is_mapping_list(messages):
        return False
    return _is_valid_dialogue(_normalize_messages(messages))


def _load_stream(
    dataset_name: str,
    split: str = "train",
    *,
    trust_remote_code: bool = False,
) -> HFIterableDataset:
    return load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        trust_remote_code=trust_remote_code,
    )


def build_sft_train_dataset(cfg: SFTDataConfig) -> HFIterableDataset:
    # 1) Japanese chat
    magpie = _load_stream("llm-jp/magpie-sft-v1.0")
    magpie = magpie.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    magpie = magpie.map(lambda ex: _map_magpie(ex, cfg.system_prompt))
    magpie = magpie.filter(_valid_example).take(cfg.magpie_take)

    # 2) Japanese reasoning
    jamard = _load_stream("if001/elyza_JaMARD_fork", trust_remote_code=True)
    jamard = jamard.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    jamard = jamard.map(lambda ex: _map_jamard(ex, cfg.system_prompt))
    jamard = jamard.filter(_valid_example).take(cfg.jamard_take)

    # 3) Japanese chat supplement: oasst2-33k-ja
    oasst2 = _load_stream("llm-jp/oasst2-33k-ja")
    oasst2 = oasst2.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    oasst2 = oasst2.map(lambda ex: _map_oasst2(ex, cfg.system_prompt))
    oasst2 = oasst2.filter(_valid_example).take(cfg.oasst2_take)

    # 3) English chat from Aya
    aya = _load_stream("CohereLabs/aya_dataset")
    aya = aya.filter(lambda ex: str(ex["language_code"]).lower() in {"eng", "en"})
    aya = aya.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    aya = aya.map(lambda ex: _map_aya_english(ex, cfg.system_prompt))
    aya = aya.filter(_valid_example).take(cfg.aya_en_take)

    # 4) Code
    coding = _load_stream("llm-jp/Synthetic-JP-EN-Coding-Dataset")
    coding = coding.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    coding = coding.map(lambda ex: _map_coding(ex, cfg.system_prompt))
    coding = coding.filter(_valid_example).take(cfg.coding_take)

    # Japanese pool = 8/10
    # 60k : 35k : 15k
    ja_pool = interleave_datasets(
        [magpie, jamard, oasst2],
        probabilities=[0.5455, 0.3182, 0.1363],
        seed=cfg.seed,
        stopping_strategy="first_exhausted",
    )

    # Final mix = 8 : 1 : 1
    mixed = interleave_datasets(
        [ja_pool, aya, coding],
        probabilities=[0.8, 0.1, 0.1],
        seed=cfg.seed,
        stopping_strategy="first_exhausted",
    )
    return mixed


class HFTorchIterableDataset(IterableDataset):
    def __init__(self, dataset: HFIterableDataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __iter__(self) -> Iterator[dict[str, object]]:
        for example in self.dataset:
            yield {"messages": example["messages"]}


def build_collate_fn(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
):
    def collate_fn(batch: list[Mapping[str, object]]) -> dict[str, torch.Tensor]:
        chats = [example["messages"] for example in batch]

        # tokenizer.chat_template は事前に Qwen3 互換のものを設定しておく
        texts = [
            tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False,
            )
            for chat in chats
        ]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        encoded["labels"] = labels
        return encoded

    return collate_fn


def build_sft_train_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    cfg: SFTDataConfig,
) -> DataLoader:
    hf_dataset = build_sft_train_dataset(cfg)
    torch_dataset = HFTorchIterableDataset(hf_dataset)

    return DataLoader(
        torch_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=build_collate_fn(tokenizer, cfg.max_length),
    )
