import argparse
import codecs
import sys
from collections.abc import Mapping
from typing import Any, Literal

import torch
from omegaconf import ListConfig

from hnet.models import HNetForCausalLM, load_hnet_config
from hnet.utils.tokenizers import ByteTokenizer


InferenceDType = Literal["auto", "bf16", "fp16", "fp32"]


def get_inference_dtype(
    device: str, requested_dtype: InferenceDType = "auto"
) -> torch.dtype:
    if requested_dtype == "bf16":
        if device != "cuda":
            raise ValueError("bf16 is only supported on CUDA device")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 is not supported on this CUDA device")
        return torch.bfloat16
    if requested_dtype == "fp16":
        if device != "cuda":
            raise ValueError("fp16 is only supported on CUDA device")
        return torch.float16
    if requested_dtype == "fp32":
        return torch.float32

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


def load_from_pretrained(
    model_path: str,
    model_config_path: str,
    requested_dtype: InferenceDType = "auto",
):
    """Load model from pretrained checkpoint.

    Args:
        model_path: Path to the model checkpoint (.pt file)
        model_config_path: Path to the model configuration (.json file)

    Returns:
        Loaded HNetForCausalLM model
    """
    hnet_cfg = load_hnet_config(model_config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference_dtype = get_inference_dtype(device, requested_dtype=requested_dtype)
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


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    sampled_logits = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(sampled_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        sampled_logits = sampled_logits.clone()
        sampled_logits[indices_to_remove] = -float("inf")

    probs = torch.softmax(sampled_logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def set_batch_size_offset(inference_params: Any, offset: int) -> None:
    if inference_params is None:
        return
    if hasattr(inference_params, "batch_size_offset"):
        inference_params.batch_size_offset = offset
    for attr_name in (
        "encoder_state",
        "routing_module_state",
        "main_network_state",
        "dechunk_state",
        "decoder_state",
    ):
        if hasattr(inference_params, attr_name):
            set_batch_size_offset(getattr(inference_params, attr_name), offset)


def generate_batch_tokens(
    model,
    prompts: list[str],
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> list[list[int]]:
    """Generate tokens for multiple prompts in a single batched decode loop."""
    if not prompts:
        return []

    device = next(model.parameters()).device
    inference_dtype = next(model.parameters()).dtype
    tokenizer = ByteTokenizer()
    eos_id = tokenizer.eos_idx

    encoded = tokenizer.encode(prompts, add_bos=True)
    sequences = [list(item["input_ids"]) for item in encoded]
    batch_size = len(sequences)
    max_prompt_len = max(len(seq) for seq in sequences)

    inference_cache = model.allocate_inference_cache(
        batch_size, max_prompt_len + max_tokens, dtype=inference_dtype
    )

    logits = torch.empty(
        (batch_size, tokenizer.vocab_size), device=device, dtype=inference_dtype
    )
    with torch.inference_mode():
        for row, seq in enumerate(sequences):
            input_ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            mask = torch.ones_like(input_ids, dtype=torch.bool)
            set_batch_size_offset(inference_cache, row)
            output = model.forward(input_ids, mask=mask, inference_params=inference_cache)
            logits[row] = output.logits[0, -1, :]
    set_batch_size_offset(inference_cache, 0)

    generated_tokens: list[list[int]] = [[] for _ in range(batch_size)]
    finished = [False for _ in range(batch_size)]

    for _ in range(max_tokens):
        if all(finished):
            break

        next_token_ids: list[int] = []
        for index in range(batch_size):
            if finished[index]:
                next_token_ids.append(eos_id)
                continue

            next_token_id = sample_next_token(
                logits[index], temperature=temperature, top_p=top_p
            )
            next_token_ids.append(next_token_id)

            if next_token_id == eos_id:
                finished[index] = True
                continue

            generated_tokens[index].append(next_token_id)

        if all(finished):
            break

        current_tokens = torch.tensor(
            next_token_ids, dtype=torch.long, device=device
        ).unsqueeze(1)
        with torch.inference_mode():
            output = model.step(current_tokens, inference_cache)
        logits = output.logits[:, -1, :]

    return generated_tokens


def generate(
    model,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.9,
):
    """Generate text from the model, yielding tokens as they are generated.

    Args:
        model: HNetForCausalLM model
        prompt: Input text prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Top-p sampling parameter

    Yields:
        Generated token ids (byte values) one by one
    """
    for token_id in generate_batch_tokens(
        model=model,
        prompts=[prompt],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )[0]:
        yield token_id


def stream_generate_and_print(
    model, prompt: str, max_tokens: int, temperature: float, top_p: float
) -> None:
    print(f"\033[92m{prompt}\033[0m", end="")
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    for token_id in generate(
        model,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    ):
        chunk = decoder.decode(bytes([token_id]), final=False)
        if chunk:
            print(chunk, end="", flush=True)

    tail = decoder.decode(b"", final=True)
    if tail:
        print(tail, end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate text from an H-Net model")
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
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter (default: 1.0)",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt text. Repeat this option to generate from multiple prompts.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Inference dtype (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used when multiple --prompt are provided (default: 1)",
    )
    args = parser.parse_args()

    print("Loading model...")
    try:
        model = load_from_pretrained(
            args.model_path, args.config_path, requested_dtype=args.dtype
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    if args.prompts:
        valid_prompts = [prompt.strip() for prompt in args.prompts if prompt.strip()]
        if args.batch_size <= 1:
            for index, prompt in enumerate(valid_prompts, start=1):
                print(
                    f"\n[{index}/{len(valid_prompts)}] Generating "
                    f"(max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p})"
                )
                stream_generate_and_print(
                    model,
                    prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                print()
            return

        decoder_factory = lambda: codecs.getincrementaldecoder("utf-8")(errors="replace")
        total = len(valid_prompts)
        for chunk_start in range(0, total, args.batch_size):
            chunk_prompts = valid_prompts[chunk_start : chunk_start + args.batch_size]
            tokens_list = generate_batch_tokens(
                model=model,
                prompts=chunk_prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            for local_idx, (prompt, tokens) in enumerate(zip(chunk_prompts, tokens_list), start=1):
                global_idx = chunk_start + local_idx
                print(
                    f"\n[{global_idx}/{total}] Generating "
                    f"(max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p})"
                )
                print(f"\033[92m{prompt}\033[0m", end="")
                decoder = decoder_factory()
                for token_id in tokens:
                    chunk = decoder.decode(bytes([token_id]), final=False)
                    if chunk:
                        print(chunk, end="", flush=True)
                tail = decoder.decode(b"", final=True)
                if tail:
                    print(tail, end="", flush=True)
                print()
        return

    while True:
        prompt = input("\nPrompt: ").strip()

        if not prompt:
            continue

        print(
            f"\nGenerating (max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p})"
        )

        stream_generate_and_print(
            model,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
