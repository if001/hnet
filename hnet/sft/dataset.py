from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Iterator, Mapping, Sequence

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

    # Qwen3 style control
    system_prompt: str = "You are a helpful assistant."

    # 1 epoch 分の概算 example 数
    # 8:1:1 = 120k : 15k : 15k
    tokyo_ja_take: int = 54_000
    jamard_take: int = 46_000
    oasst_take: int = 20_000
    tokyo_en_take: int = 15_000
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


def _first_user_text(messages: Sequence[ChatMessage]) -> str:
    for message in messages:
        if message["role"] == "user":
            return message["content"]
    return ""


def _looks_japanese(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text))


def _is_valid_dialogue(messages: Sequence[ChatMessage]) -> bool:
    if len(messages) < 2:
        return False
    if messages[-1]["role"] != "assistant":
        return False
    return any(m["role"] == "user" for m in messages)


def _prepend_qwen3_system(
    messages: Sequence[ChatMessage],
    think_mode: bool,
    system_prompt: str,
) -> list[ChatMessage]:
    control = "/think" if think_mode else "/no_think"

    out: list[ChatMessage] = []
    if messages and messages[0]["role"] == "system":
        merged_system = messages[0]["content"].strip()
        merged_system = f"{merged_system}\n{control}" if merged_system else control
        out.append({"role": "system", "content": merged_system})
        out.extend(messages[1:])
        return out

    system_content = f"{system_prompt}\n{control}".strip() if system_prompt else control
    out.append({"role": "system", "content": system_content})
    out.extend(messages)
    return out


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


def _extract_tokyotech_messages(example: Mapping[str, object]) -> list[ChatMessage]:
    # 公開 viewer から完全な列名が読み切れないため、
    # よくある候補を順に試す
    candidates = (
        "messages",
        "conversation",
        "conversations",
        "synthesized_multiturn_conversation",
        "multiturn_conversation",
        "chosen",
    )
    for field in candidates:
        if field in example:
            return _normalize_messages(example[field])

    raise KeyError(
        f"Could not find a supported conversation field in tokyotech example. "
        f"Available keys: {sorted(example.keys())}"
    )


def _map_tokyotech(
    example: Mapping[str, object], system_prompt: str
) -> dict[str, object]:
    messages = _extract_tokyotech_messages(example)
    messages = _prepend_qwen3_system(
        messages, think_mode=False, system_prompt=system_prompt
    )
    bucket = "ja" if _looks_japanese(_first_user_text(messages)) else "en"
    return {"messages": messages, "bucket": bucket}


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
    return {"messages": out, "bucket": "ja"}


def _map_oasst(example: Mapping[str, object], system_prompt: str) -> dict[str, object]:
    messages = _normalize_messages(example["conversations"])
    messages = _prepend_qwen3_system(
        messages, think_mode=False, system_prompt=system_prompt
    )
    return {"messages": messages, "bucket": "ja"}


def _map_coding(example: Mapping[str, object], system_prompt: str) -> dict[str, object]:
    messages = _normalize_messages(example["messages"])
    messages = _prepend_qwen3_system(
        messages, think_mode=False, system_prompt=system_prompt
    )
    return {"messages": messages, "bucket": "code"}


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
    # tokyotech-llm/lmsys-chat-1m-synth
    tokyo = _load_stream("tokyotech-llm/lmsys-chat-1m-synth")
    tokyo = tokyo.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    tokyo = tokyo.map(lambda ex: _map_tokyotech(ex, cfg.system_prompt))
    tokyo = tokyo.filter(_valid_example)

    tokyo_ja = tokyo.filter(lambda ex: ex["bucket"] == "ja").take(cfg.tokyo_ja_take)
    tokyo_en = tokyo.filter(lambda ex: ex["bucket"] == "en").take(cfg.tokyo_en_take)

    # elyza/JaMARD
    jamard = _load_stream("elyza/JaMARD", trust_remote_code=True)
    jamard = jamard.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    jamard = jamard.map(lambda ex: _map_jamard(ex, cfg.system_prompt))
    jamard = jamard.filter(_valid_example).take(cfg.jamard_take)

    # llm-jp/oasst1-21k-ja
    oasst = _load_stream("llm-jp/oasst1-21k-ja")
    oasst = oasst.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    oasst = oasst.map(lambda ex: _map_oasst(ex, cfg.system_prompt))
    oasst = oasst.filter(_valid_example).take(cfg.oasst_take)

    # llm-jp/Synthetic-JP-EN-Coding-Dataset
    coding = _load_stream("llm-jp/Synthetic-JP-EN-Coding-Dataset")
    coding = coding.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    coding = coding.map(lambda ex: _map_coding(ex, cfg.system_prompt))
    coding = coding.filter(_valid_example).take(cfg.coding_take)

    # 日本語バケット内の比率
    ja_pool = interleave_datasets(
        [tokyo_ja, jamard, oasst],
        probabilities=[0.45, 0.38, 0.17],
        seed=cfg.seed,
        stopping_strategy="first_exhausted",
    )

    # 最終 8:1:1
    mixed = interleave_datasets(
        [ja_pool, tokyo_en, coding],
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
        texts = [
            tokenizer.apply_chat_template(
                chat,  # type: ignore[arg-type]
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


# -------------------------
# usage
# -------------------------
# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
# cfg = SFTDataConfig(
#     batch_size=2,
#     max_length=2048,
#     system_prompt="You are a helpful assistant.",
# )
# train_dataloader = build_sft_train_dataloader(tokenizer, cfg)
# batch = next(iter(train_dataloader))
# print(batch["input_ids"].shape)
