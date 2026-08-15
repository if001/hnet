from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hnet.models import HNetForCausalLM, load_hnet_config


def parameter_group(name: str, *, tied_embeddings: bool) -> str:
    if name.startswith("embeddings."):
        return "embedding_head_shared" if tied_embeddings else "embedding"
    if name.startswith("lm_head."):
        return "output_head"
    if "routing_module" in name:
        return "router"
    if any(
        part in name
        for part in (
            ".encoder.",
            ".decoder.",
            ".chunk_layer.",
            ".dechunk_layer.",
            ".residual_proj.",
            ".pad_dimension",
        )
    ):
        return "hierarchy_encoder_decoder"
    return "main_network"


def account_config(config_path: Path) -> dict[str, Any]:
    config = load_hnet_config(config_path)
    model = HNetForCausalLM(config, device="cpu")
    groups: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        group = parameter_group(name, tied_embeddings=config.tie_embeddings)
        groups[group] = groups.get(group, 0) + parameter.numel()
    total = sum(groups.values())
    return {
        "config": str(config_path),
        "arch_layout": config.arch_layout,
        "vocab_size": config.vocab_size,
        "tie_embeddings": config.tie_embeddings,
        "total_parameters": total,
        "groups": groups,
        "group_percent": {
            name: 100.0 * count / total for name, count in groups.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report physical parameter allocation.")
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {"models": [account_config(path) for path in args.config]}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
