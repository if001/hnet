from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_model(payload: object) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
        return payload["model"]
    if isinstance(payload, Mapping) and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        return payload
    raise TypeError("Input checkpoint does not contain a model state dictionary")


def build_preconditioned_payload(
    payload: object, *, source_path: Path, source_sha256: str
) -> dict[str, Any]:
    source_step = payload.get("step") if isinstance(payload, Mapping) else None
    return {
        "model": extract_model(payload),
        "step": 0,
        "preconditioning": {
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "source_step": source_step,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a model-only step-0 checkpoint after a preconditioning run."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    output = build_preconditioned_payload(
        payload,
        source_path=args.input,
        source_sha256=sha256_file(args.input),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
