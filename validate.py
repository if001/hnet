import argparse
from pathlib import Path

import hnet.training.dataset_template as dataset_template
from hnet.training.validation import validate_checkpoint


TEMPLATE_CHOICES = sorted(
    name for name in dir(dataset_template) if name.startswith("SOURCES_")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained H-Net checkpoint and write validation_metrics.csv"
    )
    parser.add_argument(
        "--model-path",
        "--weight-path",
        dest="model_path",
        required=True,
        help="Path to checkpoint_step_XXXXXX.pt (or a model state_dict .pt).",
    )
    parser.add_argument(
        "--validation-dataset-template",
        "--validation-template",
        dest="validation_dataset_template",
        required=True,
        choices=TEMPLATE_CHOICES,
        help="Named SOURCES_* template from hnet.training.dataset_template.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "CSV output path (default: validation_metrics_post_training.csv "
            "next to checkpoint)."
        ),
    )
    parser.add_argument(
        "--model-config-path",
        default=None,
        help="Model config JSON (default: model_config.json next to checkpoint).",
    )
    parser.add_argument(
        "--training-config-path",
        default=None,
        help="Saved training config JSON (default: training_config.json next to checkpoint).",
    )
    parser.add_argument("--validation-max-batches", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--compression-ratio",
        action="append",
        dest="compression_ratios",
        type=float,
        help="Override target compression ratio. Repeat for each H-Net stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    output_path = Path(
        args.output_path
        or model_path.parent / "validation_metrics_post_training.csv"
    )
    validation_sources = list(
        getattr(dataset_template, args.validation_dataset_template)
    )
    validate_checkpoint(
        model_path=model_path,
        validation_sources=validation_sources,
        output_path=output_path,
        model_config_path=args.model_config_path,
        training_config_path=args.training_config_path,
        validation_max_batches=args.validation_max_batches,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        compression_ratios=args.compression_ratios,
    )


if __name__ == "__main__":
    main()
