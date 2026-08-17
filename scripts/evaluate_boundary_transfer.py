from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
import torch.nn.functional as F

from generate import load_from_pretrained
from hnet.training.boundary_interventions import prioritized_positions, utf8_safe_positions
from scripts.evaluate_boundary_interventions import (
    bootstrap_interval,
    encode_probe,
    load_probe_records,
    make_overrides,
    stage_input_positions,
)


def align_transfer_positions(
    source_positions: list[list[int]],
    target_positions: list[list[int]],
    token_ids: list[int],
) -> list[list[int]]:
    stage0_candidates = utf8_safe_positions(token_ids)
    stage0 = prioritized_positions(
        stage0_candidates,
        source_positions[0],
        len(target_positions[0]),
    )
    source_stage1_raw = set(source_positions[1])
    stage1_priorities = [
        index for index, raw_position in enumerate(stage0)
        if raw_position in source_stage1_raw
    ]
    stage1 = prioritized_positions(
        list(range(len(stage0))),
        stage1_priorities,
        len(target_positions[1]),
    )
    return [stage0, stage1]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer learned H-Net boundaries across models.")
    parser.add_argument("--source-model-path", type=Path, required=True)
    parser.add_argument("--source-config-path", type=Path, required=True)
    parser.add_argument("--target-model-path", type=Path, required=True)
    parser.add_argument("--target-config-path", type=Path, required=True)
    parser.add_argument("--probe-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--utf8-hard", action="store_true")
    parser.add_argument("--max-records", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_probe_records(args.probe_set)
    if args.max_records is not None:
        records = records[: args.max_records]
    source = load_from_pretrained(
        str(args.source_model_path), str(args.source_config_path), args.dtype
    )
    target = load_from_pretrained(
        str(args.target_model_path), str(args.target_config_path), args.dtype
    )
    device = next(target.parameters()).device
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        input_ids, labels = encode_probe(record.text, device)
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        continuation = (input_ids >= 0x80) & (input_ids <= 0xBF)
        with torch.inference_mode():
            source_result = source(
                input_ids=input_ids,
                mask=mask,
                continuation_mask=continuation if args.utf8_hard else None,
                continuation_hard=args.utf8_hard,
            )
            target_result = target(
                input_ids=input_ids,
                mask=mask,
                continuation_mask=continuation if args.utf8_hard else None,
                continuation_hard=args.utf8_hard,
            )
        source_outputs = list(source_result.bpred_output)
        target_outputs = list(target_result.bpred_output)
        if len(source_outputs) != 2 or len(target_outputs) != 2:
            raise ValueError("Boundary transfer requires two-stage H-Net models")
        source_positions = stage_input_positions(source_outputs)
        target_positions = stage_input_positions(target_outputs)
        transferred_positions = align_transfer_positions(
            source_positions,
            target_positions,
            [int(value) for value in input_ids[0].tolist()],
        )
        overrides = make_overrides(target_outputs, transferred_positions)
        with torch.inference_mode():
            transferred = target(
                input_ids=input_ids,
                mask=mask,
                continuation_mask=continuation if args.utf8_hard else None,
                continuation_hard=args.utf8_hard,
                boundary_overrides=overrides,
            )
        learned_nll = F.cross_entropy(
            target_result.logits[0].float(), labels[0], reduction="sum"
        )
        transfer_nll = F.cross_entropy(
            transferred.logits[0].float(), labels[0], reduction="sum"
        )
        raw_bytes = len(record.text.encode("utf-8"))
        learned_bpb = float(learned_nll.item()) / (raw_bytes * math.log(2.0))
        transfer_bpb = float(transfer_nll.item()) / (raw_bytes * math.log(2.0))
        rows.append(
            {
                "probe_id": record.probe_id,
                "category": record.category,
                "raw_bytes": raw_bytes,
                "target_learned_bpb": learned_bpb,
                "transferred_bpb": transfer_bpb,
                "delta_bpb": transfer_bpb - learned_bpb,
                "source_stage0_count": len(source_positions[0]),
                "target_stage0_count": len(target_positions[0]),
                "source_stage1_count": len(source_positions[1]),
                "target_stage1_count": len(target_positions[1]),
            }
        )
        print(f"[{index + 1}/{len(records)}] {record.probe_id}", flush=True)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(float(row["delta_bpb"]))
        grouped["__overall__"].append(float(row["delta_bpb"]))
    summary: list[dict[str, Any]] = []
    for category, values in sorted(grouped.items()):
        low, high = bootstrap_interval(values)
        summary.append(
            {
                "category": category,
                "records": len(values),
                "mean_delta_bpb": mean(values),
                "std_delta_bpb": stdev(values) if len(values) > 1 else 0.0,
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "boundary_transfer_records.csv", rows)
    write_csv(args.output_dir / "boundary_transfer_summary.csv", summary)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_model_path": str(args.source_model_path),
                "source_config_path": str(args.source_config_path),
                "target_model_path": str(args.target_model_path),
                "target_config_path": str(args.target_config_path),
                "probe_set": str(args.probe_set),
                "records": len(records),
                "alignment": "preserve target counts and prioritize source raw-byte positions",
                "utf8_hard": args.utf8_hard,
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
