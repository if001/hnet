from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hnet.training.family_consistency import (
    build_family_pair_batch,
    load_family_consistency_pairs,
)
from inspect_chunking import load_from_pretrained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate C1 landmark consistency.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    model = load_from_pretrained(args.model_path, args.config_path)
    device = next(model.parameters()).device
    pairs, manifest = load_family_consistency_pairs(
        args.data, split=args.split, seed=args.seed
    )
    rows = []
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for pair in pairs:
        batch = build_family_pair_batch(pair, device=device)
        output = model(
            input_ids=batch.input_ids,
            mask=batch.mask,
            continuation_mask=batch.continuation_mask,
            continuation_hard=True,
            num_last_tokens=1,
        )
        probability = output.bpred_output[0].boundary_prob[..., -1].float()
        selected = probability[
            torch.arange(2, device=device), batch.landmark_positions
        ]
        row = {
            "id": pair.pair_id,
            "category": pair.category,
            "left_probability": float(selected[0]),
            "right_probability": float(selected[1]),
            "absolute_difference": float(torch.abs(selected[0] - selected[1])),
            "mse": float(torch.square(selected[0] - selected[1])),
            "mean_probability": float(selected.mean()),
        }
        rows.append(row)
        grouped[pair.category].append(row)
    category = {
        name: {
            "pairs": len(values),
            "mean_absolute_difference": mean(v["absolute_difference"] for v in values),
            "mean_mse": mean(v["mse"] for v in values),
            "mean_landmark_probability": mean(v["mean_probability"] for v in values),
        }
        for name, values in sorted(grouped.items())
    }
    payload = {
        "dataset_manifest": manifest,
        "summary": {
            "pairs": len(rows),
            "mean_absolute_difference": mean(v["absolute_difference"] for v in rows),
            "mean_mse": mean(v["mse"] for v in rows),
            "mean_landmark_probability": mean(v["mean_probability"] for v in rows),
        },
        "category": category,
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
