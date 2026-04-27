import argparse
import time
from math import ceil

import torch
from torch.utils.data import DataLoader

import hnet.training.dataset_template as dataset_template
from hnet.training import DatasetSource
from hnet.training.data import DefaultRecordFormatter, StreamingByteDataset


TEMPLATE_CHOICES = sorted(
    name for name in dir(dataset_template) if name.startswith("SOURCES_")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate one-epoch micro-batches and optimizer steps for streaming datasets"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Hugging Face dataset name. Repeat to specify multiple datasets.",
    )
    parser.add_argument(
        "--dataset-template",
        type=str,
        choices=TEMPLATE_CHOICES,
        help="Named dataset template from hnet.training.dataset_template.",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle-buffer-size", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


def resolve_datasets(args: argparse.Namespace) -> list[DatasetSource]:
    if args.datasets:
        return [DatasetSource(name=name) for name in args.datasets]
    if args.dataset_template:
        return list(getattr(dataset_template, args.dataset_template))
    return [
        DatasetSource(name="if001/bunpo_phi4_ctx"),
        DatasetSource(name="if001/bunpo_phi4"),
    ]


def main() -> None:
    args = parse_args()
    sources = resolve_datasets(args)

    dataset = StreamingByteDataset(
        sources=sources,
        seq_len=args.seq_len,
        formatter=DefaultRecordFormatter(),
        shuffle_buffer_size=args.shuffle_buffer_size,
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

    print("estimating_epoch_steps=true")
    print(f"datasets={[s.name for s in sources]}")
    print(
        f"config seq_len={args.seq_len} batch_size={args.batch_size} "
        f"grad_accum_steps={args.grad_accum_steps} num_workers={args.num_workers}"
    )

    start = time.time()
    micro_batches = 0
    samples = 0
    input_tokens_bytes = 0

    for micro_batches, batch in enumerate(dataloader, start=1):
        samples += int(batch["input_ids"].shape[0])
        input_tokens_bytes += int(batch["input_ids"].numel())
        if args.log_every > 0 and micro_batches % args.log_every == 0:
            elapsed = max(time.time() - start, 1e-8)
            print(
                f"progress micro_batches={micro_batches} samples={samples} "
                f"input_tokens_bytes={input_tokens_bytes} elapsed_sec={elapsed:.1f}"
            )

    optimizer_steps = ceil(micro_batches / args.grad_accum_steps) if micro_batches > 0 else 0
    elapsed = time.time() - start

    print(f"epoch_micro_batches={micro_batches}")
    print(f"epoch_samples={samples}")
    print(f"epoch_input_tokens_bytes={input_tokens_bytes}")
    print(f"epoch_optimizer_steps={optimizer_steps}")
    print(f"elapsed_sec={elapsed:.1f}")


if __name__ == "__main__":
    main()
