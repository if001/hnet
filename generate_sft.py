import argparse
import sys
from pathlib import Path

from generate import load_from_pretrained, stream_generate_and_print
from hnet.training.data import DefaultRecordFormatter


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
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Inference dtype for model load (default: auto).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt used for chat-style formatting.",
    )
    parser.add_argument(
        "--think-mode",
        action="store_true",
        help="Use '/think' control tag instead of '/no_think'.",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Disable chat-style formatting and use prompt text as-is.",
    )
    parser.add_argument(
        "--show-formatted-prompt",
        action="store_true",
        help="Print the formatted prompt before generation.",
    )
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


def build_chat_prompt(
    user_prompt: str,
    system_prompt: str,
    think_mode: bool,
) -> str:
    control = "/think" if think_mode else "/no_think"
    system_content = f"{system_prompt}\n{control}".strip()
    formatter = DefaultRecordFormatter()
    rendered = formatter.format_record(
        {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
            ]
        }
    )
    if rendered is None:
        return f"assistant: "
    return f"{rendered}\nassistant: "


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
        model = load_from_pretrained(
            str(model_path),
            str(config_path),
            requested_dtype=args.dtype,
        )
        print("Model loaded successfully.")
    except Exception as exc:
        print(f"Error loading model: {exc}")
        sys.exit(1)

    if args.prompts:
        valid_prompts = [prompt.strip() for prompt in args.prompts if prompt.strip()]
        for idx, prompt in enumerate(valid_prompts, start=1):
            inference_prompt = (
                prompt
                if args.raw_prompt
                else build_chat_prompt(
                    prompt,
                    system_prompt=args.system_prompt,
                    think_mode=args.think_mode,
                )
            )
            print(
                f"\n[{idx}/{len(valid_prompts)}] Generating "
                f"(max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p})"
            )
            if args.show_formatted_prompt:
                print("formatted_prompt:")
                print(inference_prompt)
            stream_generate_and_print(
                model,
                inference_prompt,
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
        inference_prompt = (
            prompt
            if args.raw_prompt
            else build_chat_prompt(
                prompt,
                system_prompt=args.system_prompt,
                think_mode=args.think_mode,
            )
        )

        print(
            f"\nGenerating (max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p})"
        )
        if args.show_formatted_prompt:
            print("formatted_prompt:")
            print(inference_prompt)
        stream_generate_and_print(
            model,
            inference_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
