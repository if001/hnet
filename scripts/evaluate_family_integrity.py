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
    parser = argparse.ArgumentParser(description="Evaluate the C2 integrity margin.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aggregate(rows: list[dict[str, float | str]]) -> dict[str, float | int]:
    return {
        "pairs": len({str(row["id"]) for row in rows}),
        "sides": len(rows),
        "mean_margin_loss": mean(float(row["margin_loss"]) for row in rows),
        "margin_satisfaction_rate": mean(
            float(row["margin_satisfaction_rate"]) for row in rows
        ),
        "mean_landmark_probability": mean(
            float(row["landmark_probability"]) for row in rows
        ),
        "mean_internal_probability": mean(
            float(row["mean_internal_probability"]) for row in rows
        ),
        "mean_landmark_internal_gap": mean(
            float(row["landmark_internal_gap"]) for row in rows
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.margin <= 0:
        raise ValueError("--margin must be positive")
    model = load_from_pretrained(args.model_path, args.config_path)
    device = next(model.parameters()).device
    pairs, manifest = load_family_consistency_pairs(
        args.data,
        split=args.split,
        seed=args.seed,
        require_protected_span=True,
    )
    rows: list[dict[str, float | str]] = []
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
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
        assert batch.protected_position_mask is not None
        for index, side in enumerate(("left", "right")):
            landmark = probability[index, batch.landmark_positions[index]]
            internal = probability[index][batch.protected_position_mask[index]]
            hinge = torch.relu(args.margin + internal - landmark)
            row: dict[str, float | str] = {
                "id": pair.pair_id,
                "category": pair.category,
                "side": side,
                "margin_loss": float(hinge.mean()),
                "margin_satisfaction_rate": float(
                    ((landmark - internal) >= args.margin).float().mean()
                ),
                "landmark_probability": float(landmark),
                "mean_internal_probability": float(internal.mean()),
                "max_internal_probability": float(internal.max()),
                "landmark_internal_gap": float(landmark - internal.mean()),
            }
            rows.append(row)
            grouped[pair.category].append(row)
    payload = {
        "dataset_manifest": manifest,
        "margin": args.margin,
        "summary": aggregate(rows),
        "category": {
            category: aggregate(values) for category, values in sorted(grouped.items())
        },
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
