"""
cl-nagoya/auto-wiki-qa を few-shot example 付きデータセットへ変換する。

出力フィールド:
- question
- answer
- example1_question
- example1_answer
- example2_question
- example2_answer
- example3_question
- example3_answer
- example4_question
- example4_answer

Usage:
  python make_auto_wiki_fewshot_fields.py \
    --num-records 10000 \
    --output auto_wiki_fewshot_4examples.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

from datasets import load_dataset


@dataclass(frozen=True)
class QAItem:
    key: str
    question: str
    answer: str


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def row_to_item(row: Dict[str, Any], add_title_prefix: bool) -> QAItem:
    query = clean_text(row.get("query", ""))
    answer = clean_text(row.get("answer", ""))
    title = clean_text(row.get("title", ""))
    passage_id = clean_text(row.get("passage_id", ""))
    url = clean_text(row.get("url", ""))

    if add_title_prefix and title:
        question = f"{title}について、{query}"
    else:
        question = query

    key = f"{passage_id}::{url}::{question}::{answer}"

    return QAItem(
        key=key,
        question=question,
        answer=answer,
    )


def is_valid_item(
    item: QAItem,
    *,
    min_question_len: int,
    max_question_len: int,
    min_answer_len: int,
    max_answer_len: int,
) -> bool:
    q = item.question
    a = item.answer

    if not q or not a:
        return False

    if q == a:
        return False

    if not (min_question_len <= len(q) <= max_question_len):
        return False

    if not (min_answer_len <= len(a) <= max_answer_len):
        return False

    # 明らかにSFT用QAとして扱いにくいものを軽く除外
    bad_patterns = [
        "出典が必要",
        "要出典",
        "曖昧さ回避",
        "この記事",
        "脚注",
    ]

    if any(pattern in q for pattern in bad_patterns):
        return False

    if "http://" in a or "https://" in a:
        return False

    if "<" in a or ">" in a:
        return False

    return True


def iter_dataset(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    ds = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,
    )

    if args.shuffle_buffer_size > 0:
        ds = ds.shuffle(
            seed=args.seed,
            buffer_size=args.shuffle_buffer_size,
        )

    return iter(ds)


def build_pool(args: argparse.Namespace) -> List[QAItem]:
    pool: List[QAItem] = []
    seen: Set[str] = set()

    for row in iter_dataset(args):
        item = row_to_item(
            row,
            add_title_prefix=args.add_title_prefix,
        )

        if item.key in seen:
            continue

        if not is_valid_item(
            item,
            min_question_len=args.min_question_len,
            max_question_len=args.max_question_len,
            min_answer_len=args.min_answer_len,
            max_answer_len=args.max_answer_len,
        ):
            continue

        pool.append(item)
        seen.add(item.key)

        if len(pool) >= args.pool_size:
            break

    if len(pool) < args.num_examples + 1:
        raise RuntimeError(f"example pool is too small: {len(pool)}")

    return pool


def sample_examples(
    *,
    rng: random.Random,
    pool: Sequence[QAItem],
    target: QAItem,
    num_examples: int,
) -> List[QAItem]:
    examples: List[QAItem] = []
    used_keys: Set[str] = {target.key}

    max_trials = num_examples * 100
    trials = 0

    while len(examples) < num_examples and trials < max_trials:
        trials += 1
        candidate = rng.choice(pool)

        if candidate.key in used_keys:
            continue

        if candidate.question == target.question:
            continue

        if candidate.answer == target.answer:
            continue

        examples.append(candidate)
        used_keys.add(candidate.key)

    if len(examples) < num_examples:
        raise RuntimeError(f"failed to sample {num_examples} examples")

    return examples


def make_record(
    target: QAItem,
    examples: Sequence[QAItem],
) -> Dict[str, str]:
    record: Dict[str, str] = {
        "question": target.question,
        "answer": target.answer,
    }

    for i, example in enumerate(examples, start=1):
        record[f"example{i}_question"] = example.question
        record[f"example{i}_answer"] = example.answer

    return record


def generate_records(args: argparse.Namespace) -> Iterable[Dict[str, str]]:
    rng = random.Random(args.seed)

    pool = build_pool(args)
    seen_targets: Set[str] = set()
    count = 0

    # target側はpool作成時と別seedでshuffleする
    target_args = argparse.Namespace(**vars(args))
    target_args.seed = args.seed + 1

    for row in iter_dataset(target_args):
        target = row_to_item(
            row,
            add_title_prefix=args.add_title_prefix,
        )

        if target.key in seen_targets:
            continue

        if not is_valid_item(
            target,
            min_question_len=args.min_question_len,
            max_question_len=args.max_question_len,
            min_answer_len=args.min_answer_len,
            max_answer_len=args.max_answer_len,
        ):
            continue

        examples = sample_examples(
            rng=rng,
            pool=pool,
            target=target,
            num_examples=args.num_examples,
        )

        yield make_record(target, examples)

        seen_targets.add(target.key)
        count += 1

        if args.num_records is not None and count >= args.num_records:
            break


def write_jsonl(
    output_path: Path,
    records: Iterable[Dict[str, str]],
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="cl-nagoya/auto-wiki-qa",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("auto_wiki_fewshot_4examples.jsonl"),
    )

    parser.add_argument(
        "--num-records",
        type=int,
        default=10000,
        help="作成するレコード数",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=4,
        help="各レコードに付与するexample数",
    )

    parser.add_argument(
        "--pool-size",
        type=int,
        default=50000,
        help="example候補として保持するQA数",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--add-title-prefix",
        action="store_true",
        help="質問に '{title}について、' を付ける",
    )

    parser.add_argument("--min-question-len", type=int, default=4)
    parser.add_argument("--max-question-len", type=int, default=180)
    parser.add_argument("--min-answer-len", type=int, default=1)
    parser.add_argument("--max-answer-len", type=int, default=300)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    count = write_jsonl(
        args.output,
        generate_records(args),
    )

    print(f"wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
