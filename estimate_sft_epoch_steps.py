import argparse
import time
from math import ceil

import torch
from torch.utils.data import DataLoader

from hnet.sft.data import StreamingSFTByteDataset


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = StreamingSFTByteDataset(
        seq_len=args.context_len,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
        chat_tokenizer_path=args.chat_tokenizer_path,
    )
    dataloader_kwargs: dict[str, object] = {}
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        **dataloader_kwargs,
    )

    print("estimating_sft_epoch_steps=true")
    print(
        f"config context_len={args.context_len} batch_size={args.batch_size} "
        f"grad_accum_steps={args.grad_accum_steps} num_workers={args.num_workers} "
        f"shuffle_buffer_size={args.shuffle_buffer_size} seed={args.seed} "
        f"chat_tokenizer_path={args.chat_tokenizer_path}"
    )
    print("dataset_source=hnet/sft/dataset.py (sample mix)")

    start = time.time()
    micro_batches = 0
    samples = 0

    for micro_batches, batch in enumerate(dataloader, start=1):
        samples += int(batch["input_ids"].shape[0])
        if args.log_every > 0 and micro_batches % args.log_every == 0:
            elapsed = max(time.time() - start, 1e-8)
            print(
                f"progress micro_batches={micro_batches} samples={samples} "
                f"elapsed_sec={elapsed:.1f}"
            )

    optimizer_steps = (
        ceil(micro_batches / args.grad_accum_steps) if micro_batches > 0 else 0
    )
    elapsed = time.time() - start

    print(f"epoch_micro_batches={micro_batches}")
    print(f"epoch_samples={samples}")
    print(f"epoch_optimizer_steps={optimizer_steps}")
    print(f"recommended_max_steps={optimizer_steps}")
    print(f"elapsed_sec={elapsed:.1f}")


if __name__ == "__main__":
    main()
