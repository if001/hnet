# ============================================================
# 1. Install
# ============================================================
# !pip install -q -U datasets transformers accelerate bitsandbytes pandas pyarrow tqdm

# ============================================================
# 2. Config
# ============================================================
import os
import re
import json
import random
from pathlib import Path

import torch
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


DATASET_NAME = "cl-nagoya/auto-wiki-qa-nemotron"

MODEL_NAME = "Qwen/Qwen3-4B-Instruct"
MODEL_NAME = "Qwen/Qwen3-32B"
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

ANSWER_MAX_CHARS = 30
SAMPLE_SIZE = 20_000
SEED = 42

BATCH_SIZE = 8
MAX_NEW_TOKENS = 128
MAX_RETRIES = 2

OUT_DIR = Path("/content/auto_wiki_qa_wrong_answers")
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATH = OUT_DIR / "auto_wiki_qa_nemotron_wrong_answers_10k.jsonl"
FAILED_PATH = OUT_DIR / "failed_rows.jsonl"
PARQUET_PATH = OUT_DIR / "auto_wiki_qa_nemotron_wrong_answers_10k.parquet"


# ============================================================
# 3. Load and sample dataset
# ============================================================


def is_short_answer(example):
    ans = example.get("answer")
    if ans is None:
        return False
    ans = str(ans).strip()
    if not ans:
        return False
    return len(ans) <= ANSWER_MAX_CHARS


def _load_dataset(ds_name):
    ds = load_dataset(ds_name, split="train")
    filtered = ds.filter(is_short_answer)

    print("original rows:", len(ds))
    print(f"answer <= {ANSWER_MAX_CHARS} chars:", len(filtered))

    # row_id を固定してから shuffle する
    filtered = filtered.add_column("row_id", list(range(len(filtered))))
    sampled = filtered.shuffle(seed=SEED).select(range(min(SAMPLE_SIZE, len(filtered))))

    # 必要列だけに絞る
    items = []
    for ex in sampled:
        items.append(
            {
                "row_id": int(ex["row_id"]),
                "query": str(ex["query"]).strip(),
                "answer": str(ex["answer"]).strip(),
            }
        )

    print(len(items), items[0])
    return items, ds


# ============================================================
# 4. Load LLM
# ============================================================
def load_llm(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.eval()
    print("loaded:", MODEL_NAME)
    return tokenizer, model


# ============================================================
# 5. Prompt / generation utilities
# ============================================================
def build_messages(query: str, answer: str):
    system = (
        "あなたは日本語QAデータセットを作成するアシスタントです。"
        "与えられた質問と正解に対して、もっともらしいが誤っている短い回答を3つ作成してください。"
    )

    user = f"""次の質問に対する誤答を3つ作成してください。

# 質問
{query}

# 正解
{answer}

# 条件
- 正解と同じ回答、同義語、表記ゆれだけの回答は禁止です。
- 質問に対して明確に誤りである必要があります。
- ただし、選択肢としてはもっともらしい回答にしてください。
- 人名なら人名、地名なら地名、年号なら年号のように、正解と回答タイプをできるだけ合わせてください。
- 各誤答は30文字以内にしてください。
- 説明は不要です。
- 必ず次のJSONだけを出力してください。

{{"wrong_answers":["誤答1","誤答2","誤答3"]}}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_prompt(tokenizer, messages):
    # Qwen3系では enable_thinking=False が使える場合がある
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_batch(tokenizer, model, batch_items, temperature=0.7):
    prompts = [
        render_prompt(tokenizer, build_messages(x["query"], x["answer"]))
        for x in batch_items
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    decoded = []
    for out in outputs:
        gen_ids = out[input_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        decoded.append(text.strip())

    return decoded


# ============================================================
# 6. Parse / validate
# ============================================================
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def normalize_text(s: str) -> str:
    return str(s).strip().replace("　", " ").lower()


def extract_json(text: str):
    text = text.strip()

    # ```json ... ``` 対策
    text = text.replace("```json", "").replace("```", "").strip()

    m = JSON_RE.search(text)
    if not m:
        raise ValueError("JSON object not found")

    return json.loads(m.group(0))


def validate_wrong_answers(raw_text: str, correct_answer: str):
    obj = extract_json(raw_text)

    if "wrong_answers" not in obj:
        raise ValueError("missing wrong_answers")

    wrongs = obj["wrong_answers"]
    if not isinstance(wrongs, list):
        raise ValueError("wrong_answers is not list")

    norm_correct = normalize_text(correct_answer)

    cleaned = []
    seen = set()

    for w in wrongs:
        if not isinstance(w, str):
            continue

        w = w.strip()
        if not w:
            continue
        if len(w) > ANSWER_MAX_CHARS:
            continue

        nw = normalize_text(w)

        # 正解と同一、または包含関係にあるものは除外
        if nw == norm_correct:
            continue
        if norm_correct and (norm_correct in nw or nw in norm_correct):
            continue

        if nw in seen:
            continue

        seen.add(nw)
        cleaned.append(w)

    if len(cleaned) != 3:
        raise ValueError(f"invalid number of wrong answers: {len(cleaned)}")

    return cleaned


def make_record(item, wrongs):
    return {
        "row_id": item["row_id"],
        "query": item["query"],
        "answer": item["answer"],
        "wrong_answer_1": wrongs[0],
        "wrong_answer_2": wrongs[1],
        "wrong_answer_3": wrongs[2],
    }


def append_jsonl(path: Path, rows):
    if not rows:
        return
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================
# 7. Resume support
# ============================================================
def load_done_ids(path: Path):
    if not path.exists():
        return set()
    df = pd.read_json(path, lines=True)
    if "row_id" not in df.columns:
        return set()
    return set(df["row_id"].astype(int).tolist())


# ============================================================
# 8. Main loop
# ============================================================
def main():
    items, _ = _load_dataset(DATASET_NAME)
    tokenizer, model = load_llm(MODEL_NAME)
    done_ids = load_done_ids(JSONL_PATH)
    failed_done_ids = load_done_ids(FAILED_PATH)

    print("already done:", len(done_ids))
    print("already failed:", len(failed_done_ids))

    target_items = [
        x
        for x in items
        if x["row_id"] not in done_ids and x["row_id"] not in failed_done_ids
    ]

    print("remaining:", len(target_items))

    random.seed(SEED)
    for start in tqdm(range(0, len(target_items), BATCH_SIZE)):
        batch = target_items[start : start + BATCH_SIZE]

        pending = batch
        valid_records = []
        failed_records = []

        for attempt in range(MAX_RETRIES + 1):
            if not pending:
                break

            # リトライ時は少し温度を下げる
            temperature = 0.7 if attempt == 0 else 0.3

            outputs = generate_batch(tokenizer, model, pending, temperature=temperature)

            next_pending = []
            for item, raw in zip(pending, outputs):
                try:
                    wrongs = validate_wrong_answers(raw, item["answer"])
                    valid_records.append(make_record(item, wrongs))
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        next_pending.append(item)
                    else:
                        failed_records.append(
                            {
                                "row_id": item["row_id"],
                                "query": item["query"],
                                "answer": item["answer"],
                                "error": str(e),
                                "raw_output": raw,
                            }
                        )

            pending = next_pending

        append_jsonl(JSONL_PATH, valid_records)
        append_jsonl(FAILED_PATH, failed_records)

    print("done")
    print("output:", JSONL_PATH)
    print("failed:", FAILED_PATH)

    # ============================================================
    # 9. Convert to Parquet
    # ============================================================
    df = pd.read_json(JSONL_PATH, lines=True)

    print(df.shape)
    # display(df.head())

    df.to_parquet(PARQUET_PATH, index=False)
    print("saved:", PARQUET_PATH)

    dataset = load_dataset("parquet", data_files=PARQUET_PATH)
    # dataset.push_to_hub("repository")


if __name__ == "__main__":
    main()
