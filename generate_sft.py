import argparse
import sys
from pathlib import Path

from generate import load_from_pretrained, stream_generate_and_print


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text using SFT-trained H-Net checkpoint"
    )
    parser.add_argument(
        "--sft-output-dir",
        type=str,
        default="artifacts/hnet_sft",
        help="Directory that contains sft_final_model.pt and model_config.json",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional explicit model checkpoint path (.pt).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional explicit model config path (.json).",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt text. Repeat this option to generate from multiple prompts.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = Path(args.sft_output_dir)
    model_path = (
        Path(args.model_path)
        if args.model_path is not None
        else output_dir / "sft_final_model.pt"
    )
    config_path = (
        Path(args.config_path)
        if args.config_path is not None
        else output_dir / "model_config.json"
    )
    return model_path, config_path


def main() -> None:
    args = parse_args()
    model_path, config_path = resolve_paths(args)

    print(f"model_path={model_path}")
    print(f"config_path={config_path}")

    if not model_path.exists():
        print(f"Error: model checkpoint not found: {model_path}")
        sys.exit(1)
    if not config_path.exists():
        print(f"Error: model config not found: {config_path}")
        sys.exit(1)

    print("Loading model...")
    try:
        model = load_from_pretrained(str(model_path), str(config_path))
        print("Model loaded successfully.")
    except Exception as exc:
        print(f"Error loading model: {exc}")
        sys.exit(1)

    if args.prompts:
        valid_prompts = [prompt.strip() for prompt in args.prompts if prompt.strip()]
        for idx, prompt in enumerate(valid_prompts, start=1):
            print(
                f"\n[{idx}/{len(valid_prompts)}] Generating "
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
