from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterator, Mapping, Sequence
import json
from pathlib import Path

from datasets import interleave_datasets, load_dataset
from datasets.iterable_dataset import IterableDataset as HFIterableDataset


ChatMessage = dict[str, str]


@dataclass(frozen=True)
class SFTDataConfig:
    shuffle_buffer_size: int = 10_000
    batch_size: int = 2
    max_length: int = 2048
    num_workers: int = 0
    seed: int = 42
    mix_config_path: str | None = None

    # Qwen3-style system prompt
    system_prompt: str = "You are a helpful assistant."

    # example数ベースの概算比率（必要に応じて mix_config_path で上書き）
    magpie_take: int = 50_000
    jamard_take: int = 50_000
    oasst2_take: int = 1_000
    llm_jp_instructions_take: int = 10_000
    # select_qa_take: int = 200
    hachi_qa_take: int = 1000
    few_shot_qa_take: int = 1000

    aya_en_take: int = 15_000
    coding_take: int = 15_000

    # tool mixture = totalの約5%を想定
    xlam_take: int = 6_000
    toolace_take: int = 2_500
    apigen_mt_take: int = 1_500


def _safe_probs_from_takes(takes: Sequence[int]) -> list[float]:
    values = [max(0, int(x)) for x in takes]
    total = sum(values)
    if total <= 0:
        raise ValueError(f"All take values are non-positive: {takes}")
    return [v / total for v in values]


def _interleave_nonzero(
    datasets: Sequence[HFIterableDataset],
    takes: Sequence[int],
    *,
    seed: int,
    stopping_strategy: str = "first_exhausted",
) -> HFIterableDataset:
    if len(datasets) != len(takes):
        raise ValueError("datasets and takes must have the same length")
    selected: list[HFIterableDataset] = []
    selected_takes: list[int] = []
    for ds, take in zip(datasets, takes):
        t = max(0, int(take))
        if t <= 0:
            continue
        selected.append(ds)
        selected_takes.append(t)
    if not selected:
        if not datasets:
            raise ValueError("datasets must not be empty")
        # Return an empty stream so parent mixes can safely skip this branch
        # when its corresponding take sum is 0.
        return datasets[0].take(0)
    if len(selected) == 1:
        return selected[0]
    probs = _safe_probs_from_takes(selected_takes)
    return interleave_datasets(
        selected,
        probabilities=probs,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )


def _cfg_with_mix_overrides(cfg: SFTDataConfig) -> SFTDataConfig:
    if not cfg.mix_config_path:
        return cfg
    payload = json.loads(Path(cfg.mix_config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("mix config must be a JSON object")
    print("overwrite mix config", cfg.mix_config_path)
    allowed = {
        "shuffle_buffer_size",
        "seed",
        "magpie_take",
        "jamard_take",
        "oasst2_take",
        "llm_jp_instructions_take",
        # "select_qa_take",
        "hachi_qa_take",
        "few_shot_qa_take",
        "aya_en_take",
        "coding_take",
        "xlam_take",
        "toolace_take",
        "apigen_mt_take",
    }
    updates: dict[str, int] = {}
    for key, value in payload.items():
        if key not in allowed:
            continue
        if key in {"shuffle_buffer_size", "seed"}:
            updates[key] = int(value)
        else:
            updates[key] = max(0, int(value))
    return SFTDataConfig(
        **{
            **cfg.__dict__,
            **updates,
        }
    )


def _map_toolace(
    example: Mapping[str, object], default_system_prompt: str
) -> dict[str, object]:
    raw_system = str(example.get("system", "")).strip()
    raw_conversations = example["conversations"]

    if not _is_mapping_list(raw_conversations):
        raise ValueError("ToolACE conversations must be a list of mappings")

    system_parts: list[str] = []
    if default_system_prompt.strip():
        system_parts.append(default_system_prompt.strip())
    if raw_system:
        system_parts.append(raw_system)
    system_parts.append("/no_think")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": "\n".join(system_parts).strip()}
    ]

    for item in raw_conversations:
        raw_from = str(item["from"]).strip().lower()
        value = str(item["value"]).strip()
        if not value:
            continue

        if raw_from == "user":
            messages.append({"role": "user", "content": value})
        elif raw_from == "assistant":
            messages.append({"role": "assistant", "content": value})
        elif raw_from == "tool":
            messages.append(
                {
                    "role": "user",
                    "content": f"<tool_response>\n{value}\n</tool_response>",
                }
            )
        else:
            raise ValueError(f"Unsupported ToolACE role: {raw_from}")

    return {"messages": messages}


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


def _map_xlam(example: Mapping[str, object], system_prompt: str) -> dict[str, object]:
    tools = example.get("tools", example.get("functions", []))
    query = str(example.get("query", example.get("question", ""))).strip()
    answer = example.get("answers", example.get("answer", example.get("response", "")))

    system = system_prompt.strip()
    if tools:
        tools_json = json.dumps(tools, ensure_ascii=False)
        system = f"{system}\n/no_think\n<tools>\n{tools_json}\n</tools>"
    else:
        system = f"{system}\n/no_think"

    assistant_text = str(answer).strip()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
        {"role": "assistant", "content": assistant_text},
    ]
    return {"messages": messages}


def _tool_call_text(value: str) -> str:
    return f"<tool_call>\n{value.strip()}\n</tool_call>"


def _tool_response_text(value: str) -> str:
    return f"<tool_response>\n{value.strip()}\n</tool_response>"


def _map_apigen_mt(
    example: Mapping[str, object], default_system_prompt: str
) -> dict[str, object]:
    raw_system = str(example.get("system", "")).strip()
    raw_tools = example.get("tools", [])
    raw_conversations = example["conversations"]

    if not _is_mapping_list(raw_conversations):
        raise ValueError("APIGen-MT conversations must be a list of mappings")

    system_parts: list[str] = []
    if default_system_prompt.strip():
        system_parts.append(default_system_prompt.strip())
    if raw_system:
        system_parts.append(raw_system)
    system_parts.append("/no_think")

    if raw_tools:
        tools_json = json.dumps(raw_tools, ensure_ascii=False)
        system_parts.append(f"<tools>\n{tools_json}\n</tools>")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": "\n".join(system_parts).strip()}
    ]

    for item in raw_conversations:
        raw_from = str(item.get("from", "")).strip().lower()
        value = str(item.get("value", "")).strip()
        if not value:
            continue

        if raw_from == "human":
            messages.append({"role": "user", "content": value})

        elif raw_from == "gpt":
            messages.append({"role": "assistant", "content": value})

        elif raw_from == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": _tool_call_text(value),
                }
            )

        elif raw_from == "observation":
            messages.append(
                {
                    "role": "user",
                    "content": _tool_response_text(value),
                }
            )

        else:
            raise ValueError(f"Unsupported APIGen-MT role: {raw_from}")

    return {"messages": messages}


def _map_oasst2(example: Mapping[str, object], system_prompt: str) -> dict[str, object]:
    if "conversations" in example:
        messages = _normalize_messages(example["conversations"])
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


def _map_llm_jp_instructions(
    example: Mapping[str, object], system_prompt: str
) -> dict[str, object]:
    text = example["text"]
    output = example["output"]
    user_input = f"### 指示\n質問に対して、回答を出力してください。\n\n### 入力: \n{text}\n### 応答:\n"
    messages = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": output},
    ]
    messages = _prepend_qwen3_system(
        messages,
        think_mode=False,
        system_prompt=system_prompt,
    )
    return {"messages": messages}


def _map_select_qa(
    example: Mapping[str, object], system_prompt: str
) -> dict[str, object]:
    qa = example["question"]
    choices = example["choices"]
    ans = example["answer"]
    rational = example["rationale"]

    user_input = f"与えられた問題に対して、回答として正しい選択肢を選んでください。\n\n{qa}\n選択肢: {choices}"
    output = f"<think>\n{rational}\n</think>\n{ans}"

    messages = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": output},
    ]
    messages = _prepend_qwen3_system(
        messages,
        think_mode=True,
        system_prompt=system_prompt,
    )
    return {"messages": messages}


def _map_hachi_qa(
    example: Mapping[str, object], system_prompt: str
) -> dict[str, object]:
    instruction = example["instruction"]
    input = example["input"]
    output = example["output"]

    user_input = f"### 指示\n{instruction}\n### 入力: {input}\n### 応答:\n"
    output = f"{output}"

    messages = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": output},
    ]
    messages = _prepend_qwen3_system(
        messages,
        think_mode=False,
        system_prompt=system_prompt,
    )
    return {"messages": messages}


def _format_few_shot(q, q1, a1, q2, a2, q3, a3, q4, a4):
    return f"""### 指示
質問に対して、回答を出力してください。
<examples>
<example1>
### 入力:
質問: {q1}
### 応答:
{a1}
</example1>
<example2>
### 入力:
質問: {q2}
### 応答:
{a2}
</example2>
<example3>
### 入力:
質問: {q3}
### 応答:
{a3}
</example3>
<example4>
### 入力:
質問: {q4}
### 応答:
{a4}
</example4>
</examples>

### 入力:
質問: {q}
### 応答:
"""


def _map_few_shot_qa(
    example: Mapping[str, object], system_prompt: str
) -> dict[str, object]:
    qa = example["question"]
    q1 = example["example1_question"]
    a1 = example["example1_answer"]

    q2 = example["example2_question"]
    a2 = example["example2_answer"]

    q3 = example["example3_question"]
    a3 = example["example3_answer"]

    q4 = example["example4_question"]
    a4 = example["example4_answer"]

    ans = example["answer"]

    user_input = _format_few_shot(qa, q1, a1, q2, a2, q3, a3, q4, a4)

    messages = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": ans},
    ]
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


def _check(name, ds, limit=5):
    head_5 = ds.take(limit)
    for example in head_5:
        print(name, ": ", example["messages"])


def build_sft_train_dataset(cfg: SFTDataConfig) -> HFIterableDataset:
    cfg = _cfg_with_mix_overrides(cfg)

    # 1) Japanese chat
    magpie = _load_stream("llm-jp/magpie-sft-v1.0")
    magpie = magpie.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    magpie = magpie.map(lambda ex: _map_magpie(ex, cfg.system_prompt))
    magpie = magpie.filter(_valid_example).take(cfg.magpie_take)
    _check("magpie", magpie)

    # 2) Japanese reasoning
    jamard = _load_stream("if001/elyza_JaMARD_fork")
    jamard = jamard.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    jamard = jamard.map(lambda ex: _map_jamard(ex, cfg.system_prompt))
    jamard = jamard.filter(_valid_example).take(cfg.jamard_take)
    _check("jamard", jamard)

    # 3) Japanese chat supplement: oasst2-33k-ja
    oasst2 = _load_stream("llm-jp/oasst2-33k-ja")
    oasst2 = oasst2.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    oasst2 = oasst2.map(lambda ex: _map_oasst2(ex, cfg.system_prompt))
    oasst2 = oasst2.filter(_valid_example).take(cfg.oasst2_take)
    _check("oasst2", oasst2)

    llm_jp_instructions = _load_stream("llm-jp/llm-jp-instructions")
    llm_jp_instructions = llm_jp_instructions.shuffle(
        buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed
    )
    llm_jp_instructions = llm_jp_instructions.map(
        lambda ex: _map_llm_jp_instructions(ex, cfg.system_prompt),
        remove_columns=list(llm_jp_instructions.features.keys()),
    )
    llm_jp_instructions = llm_jp_instructions.filter(_valid_example).take(
        cfg.llm_jp_instructions_take
    )
    _check("llm_jp_instructions", llm_jp_instructions)

    # ## qa
    # select_qa = _load_stream("llm-jp/llm-jp-instructions-jculture-mcq")
    # select_qa = select_qa.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    # select_qa = select_qa.map(
    #     lambda ex: _map_select_qa(ex, cfg.system_prompt),
    #     remove_columns=list(select_qa.features.keys()),
    # )
    # select_qa = select_qa.filter(_valid_example).take(cfg.select_qa_take)
    # _check("select_qa", select_qa)

    ## hachi qa
    hachi_qa = _load_stream("HachiML/Hachi-Alpaca", split="v1.0_cleaned")
    hachi_qa = hachi_qa.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    hachi_qa = hachi_qa.map(
        lambda ex: _map_hachi_qa(ex, cfg.system_prompt),
        remove_columns=list(hachi_qa.features.keys()),
    )
    hachi_qa = hachi_qa.filter(_valid_example).take(cfg.hachi_qa_take)
    _check("hachi_qa", hachi_qa)

    ## auto-qa-few-shot
    few_shot_qa = _load_stream("if001/auto-wiki-qa-few-shot")
    few_shot_qa = few_shot_qa.shuffle(
        buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed
    )
    few_shot_qa = few_shot_qa.map(
        lambda ex: _map_few_shot_qa(ex, cfg.system_prompt),
        remove_columns=list(few_shot_qa.features.keys()),
    )
    few_shot_qa = few_shot_qa.filter(_valid_example).take(cfg.few_shot_qa_take)
    _check("few_shot_qa", few_shot_qa)

    # 3) English chat from Aya
    aya = _load_stream("CohereLabs/aya_dataset")
    aya = aya.filter(lambda ex: str(ex["language_code"]).lower() in {"eng", "en"})
    aya = aya.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    aya = aya.map(lambda ex: _map_aya_english(ex, cfg.system_prompt))
    aya = aya.filter(_valid_example).take(cfg.aya_en_take)
    _check("aya", aya)

    # 4) Code
    coding = _load_stream("llm-jp/Synthetic-JP-EN-Coding-Dataset")
    coding = coding.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    coding = coding.map(lambda ex: _map_coding(ex, cfg.system_prompt))
    coding = coding.filter(_valid_example).take(cfg.coding_take)
    # _check("coding", coding)

    xlam = _load_stream("Salesforce/xlam-function-calling-60k")
    xlam = xlam.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    xlam = xlam.map(
        lambda ex: _map_xlam(ex, cfg.system_prompt),
        remove_columns=list(xlam.features.keys()),
    )
    xlam = xlam.filter(_valid_example).take(cfg.xlam_take)
    # _check("xlam", xlam)

    toolace = _load_stream("Team-ACE/ToolACE")
    toolace = toolace.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    toolace = toolace.map(
        lambda ex: _map_toolace(ex, cfg.system_prompt),
        remove_columns=list(toolace.features.keys()),
    )
    toolace = toolace.filter(_valid_example).take(cfg.toolace_take)
    # _check("toolace", toolace)

    apigen_mt = _load_stream("Salesforce/APIGen-MT-5k")
    apigen_mt = apigen_mt.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    apigen_mt = apigen_mt.map(
        lambda ex: _map_apigen_mt(ex, cfg.system_prompt),
        remove_columns=list(apigen_mt.features.keys()),
    )
    apigen_mt = apigen_mt.filter(_valid_example).take(cfg.apigen_mt_take)

    tool_pool = _interleave_nonzero(
        [xlam, toolace, apigen_mt],
        [cfg.xlam_take, cfg.toolace_take, cfg.apigen_mt_take],
        seed=cfg.seed,
    )

    ja_pool = _interleave_nonzero(
        [magpie, jamard, oasst2, llm_jp_instructions, hachi_qa, few_shot_qa],
        [
            cfg.magpie_take,
            cfg.jamard_take,
            cfg.oasst2_take,
            cfg.llm_jp_instructions_take,
            cfg.hachi_qa_take,
            cfg.few_shot_qa_take,
        ],
        seed=cfg.seed,
    )

    mixed = _interleave_nonzero(
        [ja_pool, aya, coding, tool_pool],
        [
            cfg.magpie_take
            + cfg.jamard_take
            + cfg.oasst2_take
            + cfg.llm_jp_instructions_take
            + cfg.hachi_qa_take
            + cfg.few_shot_qa_take,
            cfg.aya_en_take,
            cfg.coding_take,
            cfg.xlam_take + cfg.toolace_take + cfg.apigen_mt_take,
        ],
        seed=cfg.seed,
    )
    return mixed
