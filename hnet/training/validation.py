import json
import logging
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .config import DatasetSource, TrainingConfig
from .trainer import (
    ValidationMetricsLogger,
    build_cached_validation_batches,
    configure_logging,
    create_dataloader,
    create_model,
    evaluate_validation,
    extract_model_state_dict,
    get_training_dtype,
    load_checkpoint_file,
    set_seed,
)


def _dataset_sources(value: object) -> list[DatasetSource]:
    if not isinstance(value, list):
        return []

    sources: list[DatasetSource] = []
    source_fields = {field.name for field in fields(DatasetSource)}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        kwargs = {key: item[key] for key in source_fields if key in item}
        sources.append(DatasetSource(**kwargs))
    return sources


def load_saved_training_config(path: str | Path) -> TrainingConfig:
    """Load the JSON emitted by train(), tolerating fields from newer versions."""
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid training config: {config_path}")

    config_fields = {field.name for field in fields(TrainingConfig)}
    kwargs: dict[str, Any] = {
        key: value for key, value in payload.items() if key in config_fields
    }
    kwargs["datasets"] = _dataset_sources(payload.get("datasets"))
    raw_validation_sources = payload.get("validation_datasets")
    kwargs["validation_datasets"] = (
        _dataset_sources(raw_validation_sources)
        if raw_validation_sources is not None
        else None
    )
    return TrainingConfig(**kwargs)


def _checkpoint_step(checkpoint: object) -> int:
    if isinstance(checkpoint, Mapping):
        step = checkpoint.get("step")
        if isinstance(step, int):
            return step
    return 0


def validate_checkpoint(
    *,
    model_path: str | Path,
    validation_sources: Sequence[DatasetSource],
    output_path: str | Path,
    model_config_path: str | Path | None = None,
    training_config_path: str | Path | None = None,
    validation_max_batches: int | None = None,
    seq_len: int | None = None,
    batch_size: int | None = None,
    compression_ratios: Sequence[float] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(model_path)
    artifact_dir = checkpoint_path.parent
    resolved_model_config_path = Path(
        model_config_path or artifact_dir / "model_config.json"
    )
    resolved_training_config_path = Path(
        training_config_path or artifact_dir / "training_config.json"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not resolved_model_config_path.is_file():
        raise FileNotFoundError(
            "Model config not found. Place model_config.json next to the checkpoint "
            "or specify --model-config-path."
        )

    if resolved_training_config_path.is_file():
        config = load_saved_training_config(resolved_training_config_path)
    else:
        config = TrainingConfig(model_config_path=str(resolved_model_config_path))

    config = replace(
        config,
        model_config_path=str(resolved_model_config_path),
        datasets=list(validation_sources),
        validation_datasets=list(validation_sources),
        packed_data_dir=None,
        packed_validation_data_dir=None,
        resume_from_checkpoint=str(checkpoint_path),
        validation_max_batches=(
            validation_max_batches
            if validation_max_batches is not None
            else config.validation_max_batches
        ),
        seq_len=seq_len if seq_len is not None else config.seq_len,
        batch_size=batch_size if batch_size is not None else config.batch_size,
        compression_ratios=(
            list(compression_ratios)
            if compression_ratios is not None
            else config.compression_ratios
        ),
    )
    if config.validation_max_batches <= 0:
        raise ValueError("validation_max_batches must be greater than zero")
    if not config.datasets:
        raise ValueError("validation_sources must not be empty")

    log = logger or configure_logging()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_dtype = get_training_dtype(device)
    log.info("device=%s validation_dtype=%s", device, training_dtype)
    log.info("validation_checkpoint=%s", checkpoint_path)
    log.info("validation_model_config=%s", resolved_model_config_path)
    if resolved_training_config_path.is_file():
        log.info("validation_training_config=%s", resolved_training_config_path)
    else:
        log.warning(
            "training_config.json was not found; using TrainingConfig defaults. "
            "Use CLI overrides if the training settings differed."
        )

    model, _ = create_model(config, device, training_dtype)
    checkpoint = load_checkpoint_file(str(checkpoint_path), device)
    model.load_state_dict(extract_model_state_dict(checkpoint), strict=True)
    model.eval()

    validation_dataloader = create_dataloader(
        config,
        sources=config.datasets,
        shuffle=False,
        num_workers=0,
    )
    validation_batches = build_cached_validation_batches(
        validation_dataloader,
        config.validation_max_batches,
    )
    metrics = evaluate_validation(
        model=model,
        training_config=config,
        device=device,
        training_dtype=training_dtype,
        validation_batches=validation_batches,
    )
    if metrics is None:
        raise RuntimeError("No validation batches could be evaluated")

    step = _checkpoint_step(checkpoint)
    metrics_logger = ValidationMetricsLogger(Path(output_path), append=False)
    metrics_logger.log(step=step, metrics=metrics)
    log.info(
        "validation step=%d batches=%d ce=%.4f bpb=%.4f csv=%s",
        step,
        metrics["validation_batches"],
        metrics["validation_ce_loss"],
        metrics["validation_bpb"],
        metrics_logger.output_path,
    )
    return metrics
