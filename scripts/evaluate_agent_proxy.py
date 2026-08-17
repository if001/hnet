from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
from tokenizers import Tokenizer

from generate import load_from_pretrained, prefill
from hnet.utils.tokenizers import ByteTokenizer


def render_prompt(task: dict[str, Any]) -> str:
    tools = json.dumps(task["tools"], ensure_ascii=False, separators=(",", ":"))
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.\n"
        "/no_think\n"
        f"<tools>\n{tools}\n</tools>"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{task['user']}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            current = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    try:
                        payload = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
                    return payload if isinstance(payload, dict) else None
        
    return None


def score_response(task: dict[str, Any], text: str) -> dict[str, Any]:
    payload = extract_first_json_object(text)
    valid_json = payload is not None
    predicted_tool = (
        payload.get("name", payload.get("tool")) if payload else None
    )
    predicted_arguments = payload.get("arguments") if payload else None
    tool_correct = predicted_tool == task["expected_tool"]
    arguments_exact = predicted_arguments == task["expected_arguments"]
    return {
        "valid_json": valid_json,
        "tool_correct": tool_correct,
        "arguments_exact": arguments_exact,
        "full_exact": valid_json and tool_correct and arguments_exact,
        "predicted_tool": predicted_tool,
        "predicted_arguments": predicted_arguments,
    }


def encode_prompt(
    prompt: str, tokenizer: Tokenizer | None
) -> tuple[list[int], int]:
    if tokenizer is None:
        byte_tokenizer = ByteTokenizer()
        return (
            byte_tokenizer.encode([prompt], add_bos=True)[0]["input_ids"].tolist(),
            byte_tokenizer.eos_idx,
        )
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer baseline requires <bos> and <eos> tokens")
    return [bos_id, *tokenizer.encode(prompt, add_special_tokens=False).ids], eos_id


def greedy_generate(
    model: Any,
    prompt_ids: list[int],
    eos_id: int,
    max_new_tokens: int,
    utf8_hard: bool,
) -> list[int]:
    device = next(model.parameters()).device
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    output, cache = prefill(model, input_ids, max_new_tokens, utf8_hard=utf8_hard)
    generated: list[int] = []
    logits = output.logits[0, -1].float()
    for _ in range(max_new_tokens):
        token = int(torch.argmax(logits).item())
        if token == eos_id:
            break
        generated.append(token)
        current = torch.tensor([[token]], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = model.step(current, cache, continuation_hard=utf8_hard)
        logits = output.logits[0, -1].float()
    return generated


def decode_generated(ids: list[int], tokenizer: Tokenizer | None) -> str:
    if tokenizer is None:
        return bytes(ids).decode("utf-8", errors="replace")
    return tokenizer.decode(ids, skip_special_tokens=True)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
        grouped["__overall__"].append(row)
    output: list[dict[str, Any]] = []
    for category, values in sorted(grouped.items()):
        output.append(
            {
                "category": category,
                "tasks": len(values),
                "json_valid_rate": mean(bool(row["valid_json"]) for row in values),
                "tool_accuracy": mean(bool(row["tool_correct"]) for row in values),
                "argument_exact_rate": mean(bool(row["arguments_exact"]) for row in values),
                "full_exact_rate": mean(bool(row["full_exact"]) for row in values),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deterministic Level A agent proxy tasks.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-tokenizer-path", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--utf8-hard", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.tasks.read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    tokenizer = (
        Tokenizer.from_file(str(args.model_tokenizer_path))
        if args.model_tokenizer_path is not None
        else None
    )
    model = load_from_pretrained(
        str(args.model_path), str(args.config_path), requested_dtype=args.dtype
    )
    rows: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "responses.jsonl").open("w", encoding="utf-8") as response_file:
        for index, task in enumerate(tasks):
            prompt = render_prompt(task)
            prompt_ids, eos_id = encode_prompt(prompt, tokenizer)
            generated = greedy_generate(
                model,
                prompt_ids,
                eos_id,
                max_new_tokens=args.max_new_tokens,
                utf8_hard=args.utf8_hard,
            )
            text = decode_generated(generated, tokenizer)
            scores = score_response(task, text)
            row = {
                "task_id": task["id"],
                "category": task["category"],
                "generated_units": len(generated),
                **scores,
            }
            row["predicted_arguments"] = json.dumps(
                row["predicted_arguments"], ensure_ascii=False, sort_keys=True
            )
            rows.append(row)
            response_file.write(
                json.dumps(
                    {"task": task, "prompt": prompt, "response": text, "scores": scores},
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"[{index + 1}/{len(tasks)}] {task['id']} {scores}", flush=True)
    summary = aggregate(rows)
    write_csv(args.output_dir / "agent_proxy_records.csv", rows)
    write_csv(args.output_dir / "agent_proxy_summary.csv", summary)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_path": str(args.model_path),
                "config_path": str(args.config_path),
                "tasks": str(args.tasks),
                "model_tokenizer_path": str(args.model_tokenizer_path) if args.model_tokenizer_path else None,
                "task_count": len(tasks),
                "max_new_tokens": args.max_new_tokens,
                "decode": "greedy",
                "prompt_format": "Qwen3 chat envelope with /no_think and <tools>",
                "accepted_tool_name_keys": ["name", "tool"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
