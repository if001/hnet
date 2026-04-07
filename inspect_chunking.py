import argparse
import sys
from collections.abc import Mapping

import torch
from omegaconf import ListConfig

from hnet.models import HNetForCausalLM, load_hnet_config
from hnet.utils.tokenizers import ByteTokenizer


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


def decode_bytes(token_ids: list[int]) -> str:
    return bytes(token_ids).decode("utf-8", errors="replace")


def render_chunk_content(token_ids: list[int]) -> str:
    """Render chunk as mixed UTF-8 text and raw bytes.

    Decodable UTF-8 spans are shown as plain text, undecodable bytes are shown as <0xHH>.
    """
    raw = bytes(token_ids)
    cursor = 0
    parts: list[str] = []

    while cursor < len(raw):
        current = raw[cursor:]
        try:
            parts.append(current.decode("utf-8"))
            break
        except UnicodeDecodeError as exc:
            if exc.start > 0:
                valid_prefix = current[: exc.start].decode("utf-8")
                parts.append(valid_prefix)

            bad = current[exc.start : exc.end]
            if bad:
                parts.extend(f"<0x{value:02X}>" for value in bad)
                cursor += exc.end
            else:
                parts.append(f"<0x{current[0]:02X}>")
                cursor += 1

    rendered = "".join(parts)
    return rendered if rendered else "<empty>"


def format_chunk_compact(token_ids: list[int]) -> str:
    mixed = render_chunk_content(token_ids)
    return f"[{mixed}]"


def format_stage_compact(chunks: list[list[int]]) -> str:
    return ", ".join(format_chunk_compact(chunk) for chunk in chunks)


def make_byte_chunks(token_ids: list[int], boundary_mask: list[bool]) -> list[list[int]]:
    boundary_positions = [idx for idx, is_boundary in enumerate(boundary_mask) if is_boundary]
    if not boundary_positions:
        return []

    chunks: list[list[int]] = []
    for i, start in enumerate(boundary_positions):
        end = boundary_positions[i + 1] if i + 1 < len(boundary_positions) else len(token_ids)
        chunks.append(token_ids[start:end])
    return chunks


def inspect_prompt(model: HNetForCausalLM, prompt: str, add_bos: bool) -> None:
    tokenizer = ByteTokenizer()
    prompt_utf8_bytes = list(prompt.encode("utf-8"))
    encoded = tokenizer.encode([prompt], add_bos=add_bos)[0]["input_ids"]
    token_ids = [int(x) for x in encoded.tolist()]

    device = next(model.parameters()).device
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)

    with torch.inference_mode():
        output = model(input_ids=input_ids, mask=mask)

    if len(output.bpred_output) < 2:
        raise ValueError(
            "This script expects a 2-stage model, but boundary outputs are fewer than 2."
        )

    stage0_mask_tensor = output.bpred_output[0].boundary_mask[0].detach().cpu()
    stage1_mask_tensor = output.bpred_output[1].boundary_mask[0].detach().cpu()

    stage0_mask = [bool(v) for v in stage0_mask_tensor.tolist()]
    stage1_mask = [bool(v) for v in stage1_mask_tensor.tolist()]

    stage0_boundaries = [idx for idx, is_boundary in enumerate(stage0_mask) if is_boundary]
    stage0_chunks = make_byte_chunks(token_ids, stage0_mask)

    stage1_source_positions = stage0_boundaries
    if len(stage1_mask) > len(stage1_source_positions):
        stage1_mask = stage1_mask[: len(stage1_source_positions)]

    stage1_boundary_indices_in_stage0 = [
        idx for idx, is_boundary in enumerate(stage1_mask) if is_boundary
    ]
    stage1_boundary_positions_in_input = [
        stage1_source_positions[idx] for idx in stage1_boundary_indices_in_stage0
    ]

    stage1_chunks: list[list[int]] = []
    if stage1_boundary_positions_in_input:
        for i, start in enumerate(stage1_boundary_positions_in_input):
            end = (
                stage1_boundary_positions_in_input[i + 1]
                if i + 1 < len(stage1_boundary_positions_in_input)
                else len(token_ids)
            )
            stage1_chunks.append(token_ids[start:end])

    print(f"Input prompt: {prompt}")
    print(f"input_prompt_utf8_bytes: {prompt_utf8_bytes}")
    print(f"model_input_token_ids: {token_ids}")
    print(f"add_bos: {add_bos}")
    print(f"input_token_count(bytes): {len(token_ids)}")
    print("\n[Stage 0]")
    print(f"boundaries(input_byte_index): {stage0_boundaries}")
    print(f"num_chunks: {len(stage0_chunks)}")

    for idx, chunk in enumerate(stage0_chunks):
        text_replace = decode_bytes(chunk)
        mixed = render_chunk_content(chunk)
        print(
            f"  - chunk{idx:03d} len={len(chunk)} bytes={chunk} "
            f"text_replace={text_replace!r} mixed={mixed!r}"
        )
    print(f"stage0: {format_stage_compact(stage0_chunks)}")

    print("\n[Stage 1]")
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
    print(f"stage1: {format_stage_compact(stage1_chunks)}")



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
            print(f"\n[{idx}/{len(valid_prompts)}]")
            inspect_prompt(model, prompt, add_bos=add_bos)
        return

    while True:
        prompt = input("\nPrompt: ").strip()
        if not prompt:
            continue
        inspect_prompt(model, prompt, add_bos=add_bos)


if __name__ == "__main__":
    main()
