from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Mapping

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hnet.models import HNetForCausalLM, load_hnet_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dense_key_for_moe_expert(key: str) -> str | None:
    marker = ".mlp.experts."
    if marker not in key:
        return None
    prefix, suffix = key.split(marker, 1)
    _expert_index, separator, expert_parameter = suffix.partition(".")
    if not separator:
        return None
    return f"{prefix}.mlp.{expert_parameter}"


def extract_model(payload: object) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
        return payload["model"]
    raise TypeError("Dense checkpoint does not contain a model state dictionary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a matched FFN-MoE step-0 state from a dense checkpoint."
    )
    parser.add_argument("--dense-checkpoint", type=Path, required=True)
    parser.add_argument("--moe-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(args.seed)
    model = HNetForCausalLM(
        load_hnet_config(args.moe_config), device="cpu", dtype=torch.bfloat16
    )
    model.init_weights()
    moe_state = model.state_dict()
    dense_payload = torch.load(
        args.dense_checkpoint, map_location="cpu", weights_only=False
    )
    dense_state = extract_model(dense_payload)

    copied_common = 0
    copied_expert = 0
    for key, target in moe_state.items():
        source_key = key if key in dense_state else dense_key_for_moe_expert(key)
        if source_key is None or source_key not in dense_state:
            continue
        source = dense_state[source_key]
        if source.shape != target.shape:
            raise ValueError(f"Shape mismatch: {source_key} -> {key}")
        target.copy_(source.to(dtype=target.dtype))
        if source_key == key:
            copied_common += 1
        else:
            copied_expert += 1

    router_keys = [key for key in moe_state if ".mlp.router." in key]
    if not router_keys or copied_expert == 0:
        raise RuntimeError("MoE config did not create routed expert parameters")
    missing_common = [
        key
        for key in dense_state
        if ".mlp." not in key and key not in moe_state
    ]
    if missing_common:
        raise RuntimeError(f"Unmapped dense non-FFN parameters: {missing_common[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": moe_state,
            "step": 0,
            "moe_initialization": {
                "dense_checkpoint": str(args.dense_checkpoint),
                "dense_checkpoint_sha256": sha256_file(args.dense_checkpoint),
                "moe_config": str(args.moe_config),
                "moe_config_sha256": sha256_file(args.moe_config),
                "seed": args.seed,
                "copied_common_tensors": copied_common,
                "copied_expert_tensors": copied_expert,
                "router_tensors": len(router_keys),
            },
        },
        args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()
