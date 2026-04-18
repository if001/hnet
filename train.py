import argparse

import hnet.training.dataset_template as dataset_template
from hnet.training import DatasetSource, TrainingConfig, train


TEMPLATE_CHOICES = sorted(
    name for name in dir(dataset_template) if name.startswith("SOURCES_")
)


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
    parser.add_argument(
        "--dataset-template",
        type=str,
        choices=TEMPLATE_CHOICES,
        help="Named dataset template from hnet.training.dataset_template.",
    )
    parser.add_argument(
        "--validation-dataset",
        action="append",
        dest="validation_datasets",
        help="Validation dataset name. Repeat to specify multiple datasets.",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--validation-max-batches", type=int, default=20)
    parser.add_argument("--validation-split-ratio", type=float, default=0.1)
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
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) to resume continued pretraining.",
    )
    parser.add_argument(
        "--no-resume-optimizer",
        action="store_true",
        help="Do not load optimizer state when resuming.",
    )
    parser.add_argument(
        "--no-resume-step",
        action="store_true",
        help="Do not restore step counter from checkpoint when resuming.",
    )
    parser.add_argument(
        "--rope-type",
        type=str,
        choices=["yarn"],
        help="Enable rope scaling type for attention (currently: yarn).",
    )
    parser.add_argument(
        "--rope-factor",
        type=float,
        help="RoPE scaling factor (e.g. 2.0, 4.0).",
    )
    parser.add_argument(
        "--rope-original-max-position-embeddings",
        type=int,
        help="Original pretraining context length for static YaRN.",
    )
    parser.add_argument(
        "--rope-attention-factor",
        type=float,
        default=None,
        help="Optional YaRN attention factor override.",
    )
    parser.add_argument("--rope-beta-fast", type=float, default=32.0)
    parser.add_argument("--rope-beta-slow", type=float, default=1.0)
    args = parser.parse_args()

    compression_ratios = args.compression_ratios or [4.0]
    lr_multipliers = args.lr_multipliers or [1.0, 1.0]

    if args.datasets:
        datasets = [DatasetSource(name=name) for name in args.datasets]
    elif args.dataset_template:
        datasets = list(getattr(dataset_template, args.dataset_template))
    else:
        datasets = [
            DatasetSource(name="if001/bunpo_phi4_ctx"),
            DatasetSource(name="if001/bunpo_phi4"),
        ]

    validation_datasets = None
    rope_scaling = None
    if args.rope_type is not None:
        if args.rope_factor is None or args.rope_original_max_position_embeddings is None:
            raise ValueError("--rope-type requires --rope-factor and --rope-original-max-position-embeddings")
        rope_scaling = {
            "rope_type": args.rope_type,
            "factor": args.rope_factor,
            "original_max_position_embeddings": args.rope_original_max_position_embeddings,
            "beta_fast": args.rope_beta_fast,
            "beta_slow": args.rope_beta_slow,
        }
        if args.rope_attention_factor is not None:
            rope_scaling["attention_factor"] = args.rope_attention_factor

    if args.validation_datasets:
        validation_datasets = [
            DatasetSource(name=name) for name in args.validation_datasets
        ]

    return TrainingConfig(
        model_config_path=args.model_config_path,
        output_dir=args.output_dir,
        datasets=datasets,
        validation_datasets=validation_datasets,
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
        validation_every=args.validation_every,
        validation_max_batches=args.validation_max_batches,
        validation_split_ratio=args.validation_split_ratio,
        train_ratio_weight=args.train_ratio_weight,
        compression_ratios=compression_ratios,
        lr_multipliers=lr_multipliers,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle_buffer_size=args.shuffle_buffer_size,
        resume_from_checkpoint=args.resume_from_checkpoint,
        resume_optimizer=not args.no_resume_optimizer,
        resume_step=not args.no_resume_step,
        rope_scaling=rope_scaling,
    )


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
