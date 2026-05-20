# eval_gsm8k_ja_openai.py
import argparse
import asyncio
import json
import os
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm


DATASET_NAME = "SakanaAI/gsm8k-ja-test_250-1319"


def build_prompt(question: str) -> str:
    return f"""以下の算数の問題を解いてください。
最後の行に「答え：<数値>」の形式で、答えの数値だけを書いてください。

問題：
{question}
"""


def normalize_text(text: str) -> str:
    # 全角数字・記号を半角へ
    return unicodedata.normalize("NFKC", text)


def parse_decimal(value: str) -> Decimal | None:
    value = normalize_text(value)
    value = value.strip()
    value = value.replace(",", "")
    value = re.sub(r"\s+", "", value)

    # 分数: 3/4 など
    if "/" in value:
        left, right = value.split("/", 1)
        try:
            return Decimal(left) / Decimal(right)
        except (InvalidOperation, ZeroDivisionError):
            return None

    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def extract_number(text: str) -> Decimal | None:
    text = normalize_text(text)

    number = r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?"

    # 優先: 明示的な答え表記
    priority_patterns = [
        rf"答え\s*[:：]\s*(?:\\boxed\{{)?\s*({number})",
        rf"回答\s*[:：]\s*(?:\\boxed\{{)?\s*({number})",
        rf"Answer\s*[:：]\s*(?:\\boxed\{{)?\s*({number})",
        rf"####\s*({number})",
        rf"\\boxed\{{\s*({number})\s*\}}",
    ]

    for pattern in priority_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            # 最後に出たものを採用
            parsed = parse_decimal(matches[-1])
            if parsed is not None:
                return parsed

    # fallback: 最後に出現した数値を採用
    matches = re.findall(number, text)
    for candidate in reversed(matches):
        parsed = parse_decimal(candidate)
        if parsed is not None:
            return parsed

    return None


async def call_model(
    client: AsyncOpenAI,
    model: str,
    question: str,
    max_tokens: int,
    timeout_retries: int,
) -> str:
    messages = [{"role": "user", "content": build_prompt(question)}]

    last_error: Exception | None = None

    for attempt in range(timeout_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            last_error = e
            await asyncio.sleep(min(2**attempt, 10))

    raise RuntimeError(f"model call failed: {last_error}")


async def eval_one(
    idx: int,
    row: dict[str, Any],
    client: AsyncOpenAI,
    args: argparse.Namespace,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    async with sem:
        gold = Decimal(str(row["answer_number"]))

        try:
            output = await call_model(
                client=client,
                model=args.model,
                question=row["question"],
                max_tokens=args.max_tokens,
                timeout_retries=args.retries,
            )
            pred = extract_number(output)
            correct = pred == gold if pred is not None else False

            return {
                "index": idx,
                "question": row["question"],
                "gold": str(gold),
                "prediction": str(pred) if pred is not None else None,
                "correct": correct,
                "output": output,
                "error": None,
            }

        except Exception as e:
            return {
                "index": idx,
                "question": row["question"],
                "gold": str(gold),
                "prediction": None,
                "correct": False,
                "output": "",
                "error": repr(e),
            }


def load_done_indices(path: Path) -> tuple[set[int], int, int]:
    if not path.exists():
        return set(), 0, 0

    done: set[int] = set()
    total = 0
    correct = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            done.add(int(item["index"]))
            total += 1
            correct += 1 if item.get("correct") else 0

    return done, total, correct


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="gsm8k_ja_results.jsonl")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=-1, help="-1 means all samples")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done, done_total, done_correct = (
        load_done_indices(out_path) if args.resume else (set(), 0, 0)
    )

    dataset = load_dataset(DATASET_NAME, split="test")

    end = len(dataset) if args.limit < 0 else min(len(dataset), args.start + args.limit)
    indices = [i for i in range(args.start, end) if i not in done]

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    sem = asyncio.Semaphore(args.concurrency)

    tasks = [eval_one(i, dataset[i], client, args, sem) for i in indices]

    total = done_total
    correct = done_correct

    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await coro
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            total += 1
            correct += 1 if result["correct"] else 0

    accuracy = correct / total if total else 0.0

    summary = {
        "dataset": DATASET_NAME,
        "model": args.model,
        "base_url": args.base_url,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "output_jsonl": str(out_path),
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
