import argparse
import sys
from collections.abc import Mapping

import torch
from omegaconf import ListConfig

from hnet.models import HNetForCausalLM, load_hnet_config
from hnet.training.chunking_utils import (
    decode_bytes,
    format_stage_compact,
    inspect_prompt_chunks,
    render_chunk_content,
)


def get_inference_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def extract_model_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        model_state = checkpoint["model"]
        if isinstance(model_state, Mapping):
            return model_state
    if isinstance(checkpoint, Mapping):
        return checkpoint
    raise TypeError("Unsupported checkpoint format")


def load_from_pretrained(model_path: str, model_config_path: str) -> HNetForCausalLM:
    hnet_cfg = load_hnet_config(model_config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference_dtype = get_inference_dtype(device)
    model = HNetForCausalLM(hnet_cfg, device=device, dtype=inference_dtype)
    model.eval()

    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 6):
        with torch.serialization.safe_globals([ListConfig]):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    else:
        checkpoint = torch.load(model_path, map_location=device)

    state_dict = extract_model_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    return model


def inspect_prompt(
    model: HNetForCausalLM,
    prompt: str,
    add_bos: bool,
    detail: bool,
    index_label: str,
) -> None:
    with torch.inference_mode():
        info = inspect_prompt_chunks(model, prompt, add_bos=add_bos)

    token_ids = info["token_ids"]
    stage0_chunks = info["stage0_chunks"]
    stage1_chunks = info["stage1_chunks"]
    prompt_utf8_bytes = info["prompt_utf8_bytes"]
    stage0_boundaries = info["stage0_boundaries"]
    stage1_boundary_indices_in_stage0 = info["stage1_boundary_indices_in_stage0"]
    stage1_boundary_positions_in_input = info["stage1_boundary_positions_in_input"]

    print(index_label)
    print(f"Input prompt: {prompt}")
    print(f"stage0: {format_stage_compact(stage0_chunks)}")
    print(f"stage1: {format_stage_compact(stage1_chunks)}")

    if not detail:
        return

    print(f"input_prompt_utf8_bytes: {prompt_utf8_bytes}")
    print(f"model_input_token_ids: {token_ids}")
    print(f"add_bos: {info['add_bos']}")
    print(f"input_token_count(bytes): {len(token_ids)}")
    print("\n[Stage 0 Details]")
    print(f"boundaries(input_byte_index): {stage0_boundaries}")
    print(f"num_chunks: {len(stage0_chunks)}")

    for idx, chunk in enumerate(stage0_chunks):
        text_replace = decode_bytes(chunk)
        mixed = render_chunk_content(chunk)
        print(
            f"  - chunk{idx:03d} len={len(chunk)} bytes={chunk} "
            f"text_replace={text_replace!r} mixed={mixed!r}"
        )

    print("\n[Stage 1 Details]")
    print(f"boundaries(stage0_chunk_index): {stage1_boundary_indices_in_stage0}")
    print(f"boundaries(input_byte_index): {stage1_boundary_positions_in_input}")
    print(f"num_chunks: {len(stage1_chunks)}")

    for idx, chunk in enumerate(stage1_chunks):
        text_replace = decode_bytes(chunk)
        mixed = render_chunk_content(chunk)
        print(
            f"  - chunk{idx:03d} len={len(chunk)} bytes={chunk} "
            f"text_replace={text_replace!r} mixed={mixed!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect stage0/stage1 chunk boundaries for a 2-stage H-Net model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the model configuration (.json file)",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt text. Repeat this option to inspect multiple prompts.",
    )
    parser.add_argument(
        "--no-bos",
        action="store_true",
        help="Disable BOS addition to the input prompt",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Show detailed chunk diagnostics in addition to compact output",
    )
    args = parser.parse_args()

    print("Loading model...")
    try:
        model = load_from_pretrained(args.model_path, args.config_path)
        print("Model loaded successfully.")
    except Exception as exc:
        print(f"Error loading model: {exc}")
        sys.exit(1)

    add_bos = not args.no_bos

    if args.prompts:
        valid_prompts = [prompt.strip() for prompt in args.prompts if prompt.strip()]
        for idx, prompt in enumerate(valid_prompts, start=1):
            if idx > 1:
                print()
            inspect_prompt(
                model,
                prompt,
                add_bos=add_bos,
                detail=args.detail,
                index_label=f"[{idx}/{len(valid_prompts)}]",
            )
        return

    prompt_count = 0
    while True:
        prompt = input("\nPrompt: ").strip()
        if not prompt:
            continue
        prompt_count += 1
        inspect_prompt(
            model,
            prompt,
            add_bos=add_bos,
            detail=args.detail,
            index_label=f"[{prompt_count}/?]",
        )


if __name__ == "__main__":
    main()
