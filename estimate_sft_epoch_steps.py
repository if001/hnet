import argparse
import time

from hnet.sft.epoch_steps import estimate_sft_epoch_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate one-epoch steps for SFT sample dataset mix defined in "
            "hnet/sft/dataset.py"
        )
    )
    parser.add_argument(
        "--context-len",
        "--seq-len",
        dest="context_len",
        type=int,
        default=512,
        help="Training sequence length (context length).",
    )
    parser.add_argument(
        "--packing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable sample packing across examples (default: true).",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle-buffer-size", type=int, default=512)
    parser.add_argument(
        "--chat-tokenizer-path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Tokenizer used for chat template rendering.",
    )
    parser.add_argument(
        "--mix-config-path",
        type=str,
        default=None,
        help="Optional JSON path to override SFT dataset take counts / seed / shuffle buffer.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("estimating_sft_epoch_steps=true")
    print(
        f"config context_len={args.context_len} batch_size={args.batch_size} "
        f"grad_accum_steps={args.grad_accum_steps} num_workers={args.num_workers} "
        f"shuffle_buffer_size={args.shuffle_buffer_size} seed={args.seed} "
        f"chat_tokenizer_path={args.chat_tokenizer_path} packing={args.packing} "
        f"mix_config_path={args.mix_config_path}"
    )
    print("dataset_source=hnet/sft/dataset.py (sample mix)")

    start = time.time()
    estimate = estimate_sft_epoch_steps(
        seq_len=args.context_len,
        packing=args.packing,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
        chat_tokenizer_path=args.chat_tokenizer_path,
        mix_config_path=args.mix_config_path,
    )
    elapsed = time.time() - start

    print(f"epoch_micro_batches={estimate.micro_batches}")
    print(f"epoch_samples={estimate.samples}")
    print(f"epoch_optimizer_steps={estimate.optimizer_steps}")
    print(f"recommended_max_steps={estimate.optimizer_steps}")
    print(f"elapsed_sec={elapsed:.1f}")


if __name__ == "__main__":
    main()
