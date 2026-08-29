from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path

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


def dense_key_for_transformer_expert(key: str) -> str | None:
    marker = ".mixer.experts.0."
    if marker not in key:
        return None
    prefix, suffix = key.split(marker, 1)
    return f"{prefix}.mixer.{suffix}"


def extract_model(payload: object) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
        return payload["model"]
    raise TypeError("Dense checkpoint does not contain a model state dictionary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a T-biased Mixer-MoE step-0 state from dense T26."
    )
    parser.add_argument("--dense-checkpoint", type=Path, required=True)
    parser.add_argument("--mixer-moe-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(args.seed)
    model = HNetForCausalLM(
        load_hnet_config(args.mixer_moe_config),
        device="cpu",
        dtype=torch.bfloat16,
    )
    model.init_weights()
    variant_state = model.state_dict()
    dense_payload = torch.load(
        args.dense_checkpoint, map_location="cpu", weights_only=False
    )
    dense_state = extract_model(dense_payload)

    copied_common = 0
    copied_transformer = 0
    copied_dense_keys: set[str] = set()
    for key, target in variant_state.items():
        source_key = (
            key if key in dense_state else dense_key_for_transformer_expert(key)
        )
        if source_key is None or source_key not in dense_state:
            continue
        source = dense_state[source_key]
        if source.shape != target.shape:
            raise ValueError(f"Shape mismatch: {source_key} -> {key}")
        target.copy_(source.to(dtype=target.dtype))
        copied_dense_keys.add(source_key)
        if source_key == key:
            copied_common += 1
        else:
            copied_transformer += 1

    missing_dense = sorted(set(dense_state) - copied_dense_keys)
    if missing_dense:
        raise RuntimeError(f"Unmapped dense parameters: {missing_dense[:5]}")
    router_keys = [key for key in variant_state if ".mixer.router." in key]
    if copied_transformer == 0 or not router_keys:
        raise RuntimeError("Mixer-MoE config did not create routed mixer parameters")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": variant_state,
            "step": 0,
            "mixer_moe_initialization": {
                "dense_checkpoint": str(args.dense_checkpoint),
                "dense_checkpoint_sha256": sha256_file(args.dense_checkpoint),
                "mixer_moe_config": str(args.mixer_moe_config),
                "mixer_moe_config_sha256": sha256_file(args.mixer_moe_config),
                "seed": args.seed,
                "copied_common_tensors": copied_common,
                "copied_transformer_tensors": copied_transformer,
                "router_tensors": len(router_keys),
            },
        },
        args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()
