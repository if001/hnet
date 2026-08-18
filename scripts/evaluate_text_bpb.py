from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from generate import load_from_pretrained
from hnet.utils.tokenizers import ByteTokenizer
from scripts.evaluate_boundary_interventions import ProbeRecord, load_probe_records


def encode_record(
    record: ProbeRecord,
    device: torch.device,
    tokenizer: Tokenizer | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if tokenizer is None:
        ids = ByteTokenizer().encode(
            [record.text], add_bos=True, add_eos=True
        )[0]["input_ids"].tolist()
    else:
        ids = tokenizer.encode(record.text, add_special_tokens=True).ids
    if len(ids) < 2:
        raise ValueError(f"Probe {record.probe_id} encoded to fewer than two IDs")
    input_ids = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    labels = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    return input_ids, labels, len(record.text.encode("utf-8"))


def bootstrap_paired_mean(
    values: list[float], seed: int = 42, samples: int = 2000
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choice(values) for _ in values) for _ in range(samples))
    return estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["category"])].append(row)
        groups["__overall__"].append(row)
    output: list[dict[str, Any]] = []
    for category, category_rows in sorted(groups.items()):
        total_nll = sum(float(row["nll_sum"]) for row in category_rows)
        total_bytes = sum(int(row["raw_bytes"]) for row in category_rows)
        per_record = [float(row["bpb"]) for row in category_rows]
        ci_low, ci_high = bootstrap_paired_mean(per_record)
        output.append(
            {
                "category": category,
                "records": len(category_rows),
                "raw_bytes": total_bytes,
                "micro_bpb": total_nll / (total_bytes * math.log(2.0)),
                "record_macro_bpb": mean(per_record),
                "record_bootstrap_95_low": ci_low,
                "record_bootstrap_95_high": ci_high,
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
    parser = argparse.ArgumentParser(description="Evaluate BPB on fixed text probes.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--probe-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-tokenizer-path", type=Path)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--utf8-hard", action="store_true")
    parser.add_argument("--max-records", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_probe_records(args.probe_set)
    if args.max_records is not None:
        records = records[: args.max_records]
    tokenizer = (
        Tokenizer.from_file(str(args.model_tokenizer_path))
        if args.model_tokenizer_path is not None
        else None
    )
    model = load_from_pretrained(
        str(args.model_path), str(args.config_path), requested_dtype=args.dtype
    )
    device = next(model.parameters()).device
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        input_ids, labels, raw_bytes = encode_record(record, device, tokenizer)
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        continuation = (input_ids >= 0x80) & (input_ids <= 0xBF)
        with torch.inference_mode():
            result = model(
                input_ids=input_ids,
                mask=mask,
                continuation_mask=continuation if args.utf8_hard else None,
                continuation_hard=args.utf8_hard,
            )
        losses = F.cross_entropy(
            result.logits[0].float(), labels[0], reduction="none"
        )
        nll_sum = float(losses.sum().item())
        rows.append(
            {
                "probe_id": record.probe_id,
                "category": record.category,
                "raw_bytes": raw_bytes,
                "model_tokens": int(labels.numel()),
                "nll_sum": nll_sum,
                "bpb": nll_sum / (raw_bytes * math.log(2.0)),
            }
        )
        print(f"[{index + 1}/{len(records)}] {record.probe_id}", flush=True)
    summary = aggregate_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "text_bpb_records.csv", rows)
    write_csv(args.output_dir / "text_bpb_summary.csv", summary)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_path": str(args.model_path),
                "config_path": str(args.config_path),
                "probe_set": str(args.probe_set),
                "model_tokenizer_path": (
                    str(args.model_tokenizer_path) if args.model_tokenizer_path else None
                ),
                "records": len(records),
                "utf8_hard": args.utf8_hard,
                "dtype": args.dtype,
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
