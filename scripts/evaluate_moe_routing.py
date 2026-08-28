from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hnet.modules.mlp import Top1SwiGLUMoE
from hnet.utils.tokenizers import ByteTokenizer
from inspect_chunking import load_from_pretrained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate FFN-MoE routing by layer and probe category."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--byte-boundary-constraint",
        choices=("off", "utf8-hard"),
        default="utf8-hard",
    )
    return parser.parse_args()


def empty_counts(num_experts: int) -> dict[str, Any]:
    return {
        "records": 0,
        "tokens": 0,
        "assignments": [0] * num_experts,
        "accepted": [0] * num_experts,
        "dropped": 0,
        "entropy_token_sum": 0.0,
    }


def add_record(
    target: dict[str, Any], module: Top1SwiGLUMoE
) -> None:
    if (
        module.last_assignment_counts is None
        or module.last_accepted_counts is None
        or module.last_routing_entropy is None
    ):
        raise RuntimeError("routing diagnostics were not populated by a forward pass")
    token_count = module.last_token_count
    target["records"] += 1
    target["tokens"] += token_count
    for index, value in enumerate(module.last_assignment_counts.tolist()):
        target["assignments"][index] += int(value)
    for index, value in enumerate(module.last_accepted_counts.tolist()):
        target["accepted"][index] += int(value)
    target["dropped"] += token_count - int(module.last_accepted_counts.sum().item())
    target["entropy_token_sum"] += float(module.last_routing_entropy) * token_count


def finalize(target: dict[str, Any]) -> dict[str, Any]:
    tokens = max(1, int(target["tokens"]))
    assignments = target["assignments"]
    return {
        "records": target["records"],
        "tokens": target["tokens"],
        "assignments": assignments,
        "assignment_fraction": [value / tokens for value in assignments],
        "max_expert_fraction": max(assignments) / tokens,
        "accepted": target["accepted"],
        "dropped": target["dropped"],
        "dropped_fraction": target["dropped"] / tokens,
        "routing_entropy": target["entropy_token_sum"] / tokens,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    model = load_from_pretrained(args.model_path, args.config_path)
    tokenizer = ByteTokenizer()
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, Top1SwiGLUMoE)
    ]
    if not modules:
        raise ValueError("model has no Top1SwiGLUMoE modules")
    num_experts = modules[0][1].num_experts
    overall = {name: empty_counts(num_experts) for name, _ in modules}
    categories: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: {name: empty_counts(num_experts) for name, _ in modules}
    )

    for record in probe["records"]:
        encoded = tokenizer.encode([record["text"]], add_bos=True)[0]["input_ids"]
        device = next(model.parameters()).device
        input_ids = encoded.to(device=device, dtype=torch.long).unsqueeze(0)
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        continuation = (input_ids >= 0x80) & (input_ids <= 0xBF)
        model(
            input_ids=input_ids,
            mask=mask,
            continuation_mask=continuation,
            continuation_hard=args.byte_boundary_constraint == "utf8-hard",
        )
        category = (
            f"family:{record['category']}"
            if record.get("family") is not None
            else f"category:{record['category']}"
        )
        for name, module in modules:
            add_record(overall[name], module)
            add_record(categories[category][name], module)

    output = {
        "version": 1,
        "model_path": args.model_path,
        "config_path": args.config_path,
        "probe": str(args.probe),
        "record_count": len(probe["records"]),
        "num_experts": num_experts,
        "layers": {name: finalize(value) for name, value in overall.items()},
        "categories": {
            category: {name: finalize(value) for name, value in layers.items()}
            for category, layers in sorted(categories.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
