from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hnet.modules.dc import BoundaryOverride
from hnet.training.chunking_utils import format_stage_compact, make_byte_chunks
from hnet.training.linguistic_boundaries import (
    FocusAnnotation,
    boundary_budget,
    boundary_positions_from_mask,
    score_focus_boundaries,
    select_topk_boundary_mask,
)
from hnet.utils.tokenizers import ByteTokenizer
from inspect_chunking import load_from_pretrained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate linguistically explainable H-Net chunk boundaries."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint-label")
    parser.add_argument(
        "--byte-boundary-constraint",
        choices=["off", "utf8-soft", "utf8-hard"],
        required=True,
    )
    parser.add_argument("--byte-boundary-constraint-bias", type=float, default=0.0)
    return parser.parse_args()


def load_probe(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") not in {1, 2, 3}:
        raise ValueError(
            "only linguistic boundary probe versions 1, 2 and 3 are supported"
        )
    if not payload.get("records") or not payload.get("budget_profiles"):
        raise ValueError("probe must contain records and budget_profiles")
    return payload


def router_tensors(output: Any, stage: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    router = output.bpred_output[stage]
    probability = router.boundary_prob[0, :, 1].detach()
    valid = router.valid_mask[0].detach().to(dtype=torch.bool)
    learned = router.boundary_mask[0].detach().to(dtype=torch.bool)
    return probability, valid, learned


def required_start_mask(valid: torch.Tensor) -> torch.Tensor:
    required = torch.zeros_like(valid, dtype=torch.bool)
    first_valid = torch.nonzero(valid, as_tuple=False)
    if not len(first_valid):
        raise ValueError("router output has no valid position")
    required[int(first_valid[0, 0].item())] = True
    return required


def stage1_input_positions(stage0_mask: torch.Tensor, stage1_mask: torch.Tensor) -> list[int]:
    stage0_positions = boundary_positions_from_mask(stage0_mask.tolist())
    usable = min(len(stage0_positions), int(stage1_mask.numel()))
    return [
        stage0_positions[index]
        for index in range(usable)
        if bool(stage1_mask[index].item())
    ]


def serialize_condition(
    text: str,
    token_ids: list[int],
    annotation: FocusAnnotation,
    stage0_mask: torch.Tensor,
    stage1_mask: torch.Tensor,
    stage0_probability: torch.Tensor,
    stage1_probability: torch.Tensor,
) -> dict[str, Any]:
    stage0_positions = boundary_positions_from_mask(stage0_mask.tolist())
    stage1_positions = stage1_input_positions(stage0_mask, stage1_mask)
    stage1_input_mask = [False] * len(token_ids)
    for position in stage1_positions:
        stage1_input_mask[position] = True
    stage0_chunks = make_byte_chunks(token_ids, stage0_mask.tolist())
    stage1_chunks = make_byte_chunks(token_ids, stage1_input_mask)
    return {
        "stage0": {
            "boundary_count": len(stage0_positions),
            "boundary_positions": stage0_positions,
            "boundary_probability": [float(value) for value in stage0_probability.cpu()],
            "chunks": format_stage_compact(stage0_chunks),
            "score": score_focus_boundaries(text, annotation, stage0_positions),
        },
        "stage1": {
            "boundary_count": len(stage1_positions),
            "boundary_positions": stage1_positions,
            "boundary_probability": [float(value) for value in stage1_probability.cpu()],
            "chunks": format_stage_compact(stage1_chunks),
            "score": score_focus_boundaries(text, annotation, stage1_positions),
        },
    }


@torch.inference_mode()
def evaluate_record(
    model: Any,
    tokenizer: ByteTokenizer,
    record: dict[str, Any],
    budget_profiles: list[dict[str, Any]],
    constraint: str,
    constraint_bias: float,
) -> dict[str, Any]:
    text = record["text"]
    encoded = tokenizer.encode([text], add_bos=True)[0]["input_ids"]
    token_ids = [int(value) for value in encoded.tolist()]
    device = next(model.parameters()).device
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    continuation = (input_ids >= 0x80) & (input_ids <= 0xBF)
    forward_kwargs = {
        "input_ids": input_ids,
        "mask": mask,
        "continuation_mask": continuation if constraint != "off" else None,
        "continuation_bias": constraint_bias if constraint == "utf8-soft" else 0.0,
        "continuation_hard": constraint == "utf8-hard",
    }
    focus = record["focus"]
    annotation = FocusAnnotation(
        surface=focus["surface"],
        occurrence=int(focus.get("occurrence", 0)),
        acceptable_segmentations=tuple(focus["acceptable_segmentations"]),
        protected_substrings=tuple(focus.get("protected_substrings", ())),
    )

    native_output = model(**forward_kwargs)
    if len(native_output.bpred_output) != 2:
        raise ValueError("linguistic boundary evaluation currently expects a 2-stage H-Net")
    stage0_probability, stage0_valid, stage0_native = router_tensors(native_output, 0)
    stage1_probability, _, stage1_native = router_tensors(native_output, 1)
    conditions = {
        "native": serialize_condition(
            text,
            token_ids,
            annotation,
            stage0_native,
            stage1_native,
            stage0_probability,
            stage1_probability,
        )
    }

    for profile in budget_profiles:
        stage0_count = boundary_budget(
            int(stage0_valid.sum().item()), float(profile["stage0_units_per_chunk"])
        )
        forced_stage0 = select_topk_boundary_mask(
            stage0_probability,
            stage0_valid,
            stage0_count,
            required_start_mask(stage0_valid),
        )
        stage0_override = BoundaryOverride(boundary_mask=forced_stage0.unsqueeze(0))
        stage0_forced_output = model(
            **forward_kwargs, boundary_overrides=[stage0_override]
        )
        forced_stage0_probability, _, actual_stage0 = router_tensors(
            stage0_forced_output, 0
        )
        forced_stage1_probability, stage1_valid, _ = router_tensors(
            stage0_forced_output, 1
        )
        stage1_count = boundary_budget(
            int(stage1_valid.sum().item()), float(profile["stage1_units_per_chunk"])
        )
        forced_stage1 = select_topk_boundary_mask(
            forced_stage1_probability,
            stage1_valid,
            stage1_count,
            required_start_mask(stage1_valid),
        )
        conditions[profile["id"]] = serialize_condition(
            text,
            token_ids,
            annotation,
            actual_stage0,
            forced_stage1,
            forced_stage0_probability,
            forced_stage1_probability,
        )

    return {
        "id": record["id"],
        "category": record["category"],
        "text": text,
        "focus": focus,
        "pair": record.get("pair"),
        "family": record.get("family"),
        "input_token_count": len(token_ids),
        "conditions": conditions,
    }


def main() -> None:
    args = parse_args()
    probe = load_probe(args.probe)
    model = load_from_pretrained(args.model_path, args.config_path)
    tokenizer = ByteTokenizer()
    records = [
        evaluate_record(
            model,
            tokenizer,
            record,
            probe["budget_profiles"],
            args.byte_boundary_constraint,
            args.byte_boundary_constraint_bias,
        )
        for record in probe["records"]
    ]
    output = {
        "version": int(probe["version"]),
        "model_name": args.model_name,
        "model_path": args.model_path,
        "config_path": args.config_path,
        "seed": args.seed,
        "checkpoint_label": args.checkpoint_label,
        "probe_path": str(args.probe),
        "byte_boundary_constraint": args.byte_boundary_constraint,
        "byte_boundary_constraint_bias": args.byte_boundary_constraint_bias,
        "budget_profiles": probe["budget_profiles"],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
