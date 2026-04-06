import argparse
import codecs
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


def load_from_pretrained(model_path: str, model_config_path: str):
    """Load model from pretrained checkpoint.

    Args:
        model_path: Path to the model checkpoint (.pt file)
        model_config_path: Path to the model configuration (.json file)

    Returns:
        Loaded HNetForCausalLM model
    """
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
    device = next(model.parameters()).device
    inference_dtype = next(model.parameters()).dtype
    tokenizer = ByteTokenizer()
    encoded = tokenizer.encode([prompt], add_bos=True)[0]
    input_ids = torch.tensor(
        encoded["input_ids"], dtype=torch.long, device=device
    ).unsqueeze(0)

    inference_cache = model.allocate_inference_cache(
        1, input_ids.shape[1] + max_tokens, dtype=inference_dtype
    )

    with torch.inference_mode():
        mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
        output = model.forward(input_ids, mask=mask, inference_params=inference_cache)

    logits = output.logits[0, -1, :] / temperature

    for _ in range(max_tokens):
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        if next_token.item() == tokenizer.eos_idx:
            break

        current_token = next_token.unsqueeze(0)
        yield int(next_token.item())

        with torch.inference_mode():
            output = model.step(current_token, inference_cache)

        logits = output.logits[0, -1, :] / temperature


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
    args = parser.parse_args()

    print("Loading model...")
    try:
        model = load_from_pretrained(args.model_path, args.config_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    if args.prompts:
        for index, prompt in enumerate(args.prompts, start=1):
            prompt = prompt.strip()
            if not prompt:
                continue

            print(
                f"\n[{index}/{len(args.prompts)}] Generating "
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
