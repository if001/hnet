import argparse

from hnet.training import DatasetSource, TrainingConfig, train


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train H-Net on Hugging Face datasets")
    parser.add_argument(
        "--model-config-path",
        type=str,
        default="configs/hnet_1stage_100m.json",
        help="Path to the model config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/hnet_1stage_100m",
        help="Directory where checkpoints will be written.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Hugging Face dataset name. Repeat to specify multiple datasets.",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--train-ratio-weight", type=float, default=0.02)
    parser.add_argument(
        "--compression-ratio",
        action="append",
        dest="compression_ratios",
        type=float,
        help="Target compression ratio per H-Net stage. Repeat for multi-stage models.",
    )
    parser.add_argument(
        "--lr-multiplier",
        action="append",
        dest="lr_multipliers",
        type=float,
        help="Learning-rate multiplier per stage. Repeat for each hierarchy level.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle-buffer-size", type=int, default=512)
    args = parser.parse_args()

    dataset_names = args.datasets or ["if001/bunpo_phi4_ctx", "if001/bunpo_phi4"]
    compression_ratios = args.compression_ratios or [4.0]
    lr_multipliers = args.lr_multipliers or [1.0, 1.0]

    return TrainingConfig(
        model_config_path=args.model_config_path,
        output_dir=args.output_dir,
        datasets=[DatasetSource(name=name) for name in dataset_names],
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        log_every=args.log_every,
        save_every=args.save_every,
        train_ratio_weight=args.train_ratio_weight,
        compression_ratios=compression_ratios,
        lr_multipliers=lr_multipliers,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
