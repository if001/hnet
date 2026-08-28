import argparse
import math
import numpy as np
import hnet.training.dataset_template as dataset_template
from hnet.models import load_hnet_config
from hnet.training import DatasetSource, TrainingConfig, train


TEMPLATE_CHOICES = sorted(
    name for name in dir(dataset_template) if name.startswith("SOURCES_")
)


def compute_default_lr_multipliers(
    compression_ratios: list[float],
    d_model: list[int],
    n_gpt: float = 4.6,
) -> list[float]:
    # https://arxiv.org/pdf/2507.07955#page=35

    multipliers = []
    for i in range(len(compression_ratios)):
        d1 = np.prod(compression_ratios[i:]) / np.prod(compression_ratios)
        d2 = d_model[-1] / d_model[i]
        multipliers.append(math.sqrt(n_gpt * d1 * d2))
    return multipliers


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
        "--packed-data-dir",
        type=str,
        default=None,
        help="Path to prepacked tokenized dataset directory (mix_manifest.json + shard files).",
    )
    parser.add_argument(
        "--packed-validation-data-dir",
        type=str,
        default=None,
        help="Optional packed validation dataset directory.",
    )
    parser.add_argument(
        "--validation-dataset",
        action="append",
        dest="validation_datasets",
        help="Validation dataset name. Repeat to specify multiple datasets.",
    )
    parser.add_argument(
        "--chunk-prompt",
        action="append",
        dest="chunk_prompts",
        help="Prompt text for chunk inspection saved during validation. Repeat to specify multiple prompts.",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--max-train-bytes",
        type=int,
        default=None,
        help="Stop after consuming at least this many raw input bytes.",
    )
    parser.add_argument(
        "--lr-schedule-steps",
        type=int,
        default=None,
        help=(
            "Optional WSD schedule horizon. This can exceed --max-steps so a "
            "short calibration uses the same learning rates as a longer run."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--save-every-bytes",
        type=int,
        default=None,
        help="Also save when cumulative raw input bytes cross this interval.",
    )
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument(
        "--validation-every-bytes",
        type=int,
        default=None,
        help="Also validate when cumulative raw input bytes cross this interval.",
    )
    parser.add_argument("--validation-max-batches", type=int, default=20)
    parser.add_argument("--validation-split-ratio", type=float, default=0.1)
    parser.add_argument("--train-ratio-weight", type=float, default=0.02)
    parser.add_argument(
        "--byte-boundary-constraint",
        type=str,
        choices=["off", "utf8-soft", "utf8-hard"],
        default="off",
        help=(
            "Optional constraint against chunk boundaries on UTF-8 continuation "
            "bytes. utf8-soft adds a prior and auxiliary loss; utf8-hard removes "
            "continuation bytes from stage-0 boundary candidates."
        ),
    )
    parser.add_argument(
        "--byte-boundary-constraint-weight",
        type=float,
        default=0.0,
        help="Auxiliary loss weight for UTF-8 continuation-byte boundary penalty.",
    )
    parser.add_argument(
        "--byte-boundary-constraint-bias",
        type=float,
        default=0.0,
        help="Soft bias strength that lowers stage0 boundary probability on UTF-8 continuation bytes.",
    )
    parser.add_argument(
        "--family-consistency-data",
        type=str,
        default=None,
        help="JSON dataset of paired landmarks used for the C1 auxiliary forward.",
    )
    parser.add_argument(
        "--family-consistency-split", type=str, default="train"
    )
    parser.add_argument(
        "--family-consistency-objective",
        choices=("c1", "c2"),
        default="c1",
        help="C1 probability matching or C2 protected-span integrity margin.",
    )
    parser.add_argument(
        "--family-consistency-weight",
        type=float,
        default=0.0,
        help="Weight of the selected family auxiliary loss. Use zero for a sham control.",
    )
    parser.add_argument(
        "--family-consistency-margin",
        type=float,
        default=0.15,
        help="Required landmark-minus-internal probability margin for C2.",
    )
    parser.add_argument("--family-consistency-seed", type=int, default=42)
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Legacy fallback used for every split seed that is not specified.",
    )
    parser.add_argument(
        "--model-init-seed",
        type=int,
        default=None,
        help="Seed used only while constructing initial model parameters.",
    )
    parser.add_argument(
        "--data-order-seed",
        type=int,
        default=None,
        help="Seed used only for packed training sample order.",
    )
    parser.add_argument(
        "--train-runtime-seed",
        type=int,
        default=None,
        help="Seed restored after model initialization for training-time RNG.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle-buffer-size", type=int, default=512)
    parser.add_argument(
        "--initial-model-checkpoint",
        type=str,
        default=None,
        help="Load an exact step-0 model state without resuming step/data/optimizer.",
    )
    parser.add_argument(
        "--save-initial-model-to",
        type=str,
        default=None,
        help="Save the exact model state used before the first optimizer step.",
    )
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
        "--freeze-mode",
        choices=["none", "router", "main", "outer"],
        default="none",
        help=(
            "Freeze routing modules (router), update only routers (main), "
            "update only outer-stage encoder/decoder/routing/dechunk modules "
            "(outer), or update the full model (none)."
        ),
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

    if args.packed_validation_data_dir is not None and args.packed_data_dir is None:
        raise ValueError("--packed-validation-data-dir requires --packed-data-dir")
    if args.initial_model_checkpoint and args.resume_from_checkpoint:
        raise ValueError(
            "--initial-model-checkpoint and --resume-from-checkpoint are mutually exclusive"
        )

    compression_ratios = args.compression_ratios or [4.0]

    if args.lr_multipliers:
        lr_multipliers = args.lr_multipliers
    else:
        model_config = load_hnet_config(args.model_config_path)
        _comp = compression_ratios + [1]
        lr_multipliers = compute_default_lr_multipliers(
            compression_ratios=_comp,
            d_model=model_config.d_model,
        )
        print("auto calc lr_multipliers", lr_multipliers)

    if args.packed_data_dir is not None:
        datasets = []
    elif args.datasets:
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
        if (
            args.rope_factor is None
            or args.rope_original_max_position_embeddings is None
        ):
            raise ValueError(
                "--rope-type requires --rope-factor and --rope-original-max-position-embeddings"
            )
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
        packed_data_dir=args.packed_data_dir,
        packed_validation_data_dir=args.packed_validation_data_dir,
        validation_datasets=validation_datasets,
        chunk_prompts=[p for p in (args.chunk_prompts or []) if p.strip()],
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        max_train_bytes=args.max_train_bytes,
        lr_schedule_steps=args.lr_schedule_steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        log_every=args.log_every,
        save_every=args.save_every,
        save_every_bytes=args.save_every_bytes,
        validation_every=args.validation_every,
        validation_every_bytes=args.validation_every_bytes,
        validation_max_batches=args.validation_max_batches,
        validation_split_ratio=args.validation_split_ratio,
        train_ratio_weight=args.train_ratio_weight,
        byte_boundary_constraint=args.byte_boundary_constraint,
        byte_boundary_constraint_weight=args.byte_boundary_constraint_weight,
        byte_boundary_constraint_bias=args.byte_boundary_constraint_bias,
        family_consistency_data=args.family_consistency_data,
        family_consistency_split=args.family_consistency_split,
        family_consistency_objective=args.family_consistency_objective,
        family_consistency_weight=args.family_consistency_weight,
        family_consistency_margin=args.family_consistency_margin,
        family_consistency_seed=args.family_consistency_seed,
        compression_ratios=compression_ratios,
        lr_multipliers=lr_multipliers,
        seed=args.seed,
        model_init_seed=args.model_init_seed,
        data_order_seed=args.data_order_seed,
        train_runtime_seed=args.train_runtime_seed,
        num_workers=args.num_workers,
        shuffle_buffer_size=args.shuffle_buffer_size,
        initial_model_checkpoint=args.initial_model_checkpoint,
        save_initial_model_to=args.save_initial_model_to,
        resume_from_checkpoint=args.resume_from_checkpoint,
        resume_optimizer=not args.no_resume_optimizer,
        resume_step=not args.no_resume_step,
        freeze_mode=args.freeze_mode,
        rope_scaling=rope_scaling,
    )


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
