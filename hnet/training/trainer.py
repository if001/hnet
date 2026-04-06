import csv
import json
import logging
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..models.config_hnet import HNetConfig
from ..models.config_io import load_hnet_config, save_hnet_config
from ..models.mixer_seq import HNetForCausalLM
from ..utils.train import group_params, load_balancing_loss
from .config import DatasetSource, TrainingConfig
from .data import DefaultRecordFormatter, StreamingByteDataset


class TrainingMetricsLogger:
    fieldnames = ["step", "learning_rate", "ce_loss", "ratio_loss", "total_loss"]

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_file()

    def _initialize_file(self) -> None:
        with self.output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(
        self,
        step: int,
        learning_rate: float,
        ce_loss: float,
        ratio_loss: float,
        total_loss: float,
    ) -> None:
        with self.output_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(
                {
                    "step": step,
                    "learning_rate": learning_rate,
                    "ce_loss": ce_loss,
                    "ratio_loss": ratio_loss,
                    "total_loss": total_loss,
                }
            )


class ValidationMetricsLogger:
    fieldnames = [
        "step",
        "validation_batches",
        "validation_ce_loss",
        "validation_bpb",
        "avg_bytes_per_chunk",
        "target_ratio_gap",
        "actual_selected_fraction",
        "mean_boundary_probability",
        "boundary_positions_sample",
        "stage0_selected_fraction",
        "stage1_selected_fraction",
        "stage0_target_ratio_gap",
        "stage1_target_ratio_gap",
        "compression_l1_l0",
        "compression_l2_l1",
        "compression_l2_l0",
    ]

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_file()

    def _initialize_file(self) -> None:
        with self.output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        payload = {"step": step, **metrics}
        with self.output_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(payload)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("hnet.train")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def format_parameter_count(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    return f"{count:,}"


def get_training_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def create_model(
    training_config: TrainingConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[HNetForCausalLM, HNetConfig]:
    config = load_hnet_config(training_config.model_config_path)
    model = HNetForCausalLM(config, device=device, dtype=dtype)
    model.init_weights()
    model.apply_lr_multiplier(training_config.lr_multipliers)
    return model, config


def create_dataloader(
    training_config: TrainingConfig,
    sources: list[DatasetSource] | None = None,
    shuffle: bool = True,
) -> DataLoader[dict[str, torch.Tensor]]:
    dataset = StreamingByteDataset(
        sources=sources or training_config.datasets,
        seq_len=training_config.seq_len,
        formatter=DefaultRecordFormatter(),
        shuffle_buffer_size=training_config.shuffle_buffer_size,
        shuffle=shuffle,
    )
    return DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_ratio_loss(
    boundary_predictions: list[object],
    compression_ratios: list[float],
    device: torch.device,
) -> torch.Tensor:
    if not boundary_predictions:
        return torch.tensor(0.0, device=device)

    losses: list[torch.Tensor] = []
    for index, router_output in enumerate(boundary_predictions):
        ratio = compression_ratios[min(index, len(compression_ratios) - 1)]
        losses.append(load_balancing_loss(router_output, ratio))
    return torch.stack(losses).mean()


def apply_weight_decay(
    param_groups: list[dict[str, object]], weight_decay: float
) -> None:
    for group in param_groups:
        if "weight_decay" not in group:
            group["weight_decay"] = weight_decay


def apply_learning_rate(optimizer: AdamW, base_learning_rate: float) -> None:
    for group in optimizer.param_groups:
        multiplier = float(group.get("lr_multiplier", 1.0))
        group["lr"] = base_learning_rate * multiplier


def wsd_schedule(
    step: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
    warmup_ratio: float = 0.1,
    decay_ratio: float = 0.2,
    inverse_sqrt_span: float = 100.0,
) -> float:
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    decay_steps = max(1, int(total_steps * decay_ratio))
    stable_end = max(warmup_steps, total_steps - decay_steps)

    if step < warmup_steps:
        return max_lr * float(step + 1) / float(warmup_steps)

    if step < stable_end:
        return max_lr

    decay_progress = (step - stable_end) / float(max(1, total_steps - stable_end))
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    end_value = 1.0 / math.sqrt(inverse_sqrt_span)
    current_value = 1.0 / math.sqrt(1.0 + decay_progress * (inverse_sqrt_span - 1.0))
    normalized = (current_value - end_value) / max(1e-8, (1.0 - end_value))
    return min_lr + (max_lr - min_lr) * normalized


def save_training_config(training_config: TrainingConfig, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(training_config), ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    return output_path


def estimate_dataset_examples(
    training_config: TrainingConfig,
    sources: list[DatasetSource] | None = None,
) -> tuple[int | None, int | None]:
    use_sources = sources or training_config.datasets
    example_counts = []
    for source in use_sources:
        if source.take_examples <= 0:
            return None, None
        example_counts.append(source.take_examples)

    if not example_counts:
        return 0, 0

    total_examples = sum(example_counts)
    optimizer_steps = total_examples // (
        training_config.batch_size * training_config.grad_accum_steps
    )
    return total_examples, optimizer_steps


def split_train_validation_sources(
    sources: list[DatasetSource],
    validation_split_ratio: float,
) -> tuple[list[DatasetSource], list[DatasetSource]]:
    if not 0.0 < validation_split_ratio < 1.0:
        raise ValueError(
            f"validation_split_ratio must be in (0, 1), got {validation_split_ratio}"
        )

    train_sources: list[DatasetSource] = []
    validation_sources: list[DatasetSource] = []

    for source in sources:
        if source.take_examples <= 1:
            raise ValueError(
                "validation split requires finite take_examples >= 2 for each dataset source"
            )

        total = source.take_examples
        val_take = max(1, int(round(total * validation_split_ratio)))
        val_take = min(val_take, total - 1)
        train_take = total - val_take

        train_sources.append(
            DatasetSource(
                name=source.name,
                split=source.split,
                config_name=source.config_name,
                take_examples=train_take,
                skip_examples=source.skip_examples,
            )
        )
        validation_sources.append(
            DatasetSource(
                name=source.name,
                split=source.split,
                config_name=source.config_name,
                take_examples=val_take,
                skip_examples=source.skip_examples + train_take,
            )
        )

    return train_sources, validation_sources


def save_checkpoint(
    model: HNetForCausalLM,
    optimizer: AdamW,
    step: int,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint_step_{step:06d}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        checkpoint_path,
    )
    return checkpoint_path


def build_cached_validation_batches(
    validation_dataloader: DataLoader[dict[str, torch.Tensor]],
    max_batches: int,
) -> list[dict[str, torch.Tensor]]:
    if max_batches <= 0:
        return []

    cached_batches: list[dict[str, torch.Tensor]] = []
    validation_iterator = iter(validation_dataloader)
    for _ in range(max_batches):
        try:
            batch = next(validation_iterator)
        except StopIteration:
            break
        cached_batches.append(batch)

    return cached_batches


def _format_boundary_positions(mask: torch.Tensor, limit: int = 64) -> str:
    indices = torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist()
    if not indices:
        return ""
    visible = indices[:limit]
    text = "|".join(str(i) for i in visible)
    if len(indices) > limit:
        text += "|..."
    return text


@torch.no_grad()
def evaluate_validation(
    model: HNetForCausalLM,
    training_config: TrainingConfig,
    device: torch.device,
    training_dtype: torch.dtype,
    validation_batches: list[dict[str, torch.Tensor]],
) -> dict[str, Any] | None:
    if training_config.validation_max_batches <= 0 or not validation_batches:
        return None

    ce_sum = 0.0
    token_count = 0
    boundary_positions_sample = ""

    selected_sum_by_stage: list[float] = []
    mean_prob_sum_by_stage: list[float] = []
    ratio_gap_sum_by_stage: list[float] = []
    count_by_stage: list[int] = []

    processed_batches = 0

    for batch in validation_batches:
        processed_batches += 1

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=training_dtype,
            enabled=device.type == "cuda",
        ):
            output = model(input_ids=input_ids, mask=mask)
            ce_loss = F.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                labels.reshape(-1),
            )

        tokens_in_batch = int(labels.numel())
        ce_sum += float(ce_loss.detach()) * tokens_in_batch
        token_count += tokens_in_batch

        if not boundary_positions_sample and output.bpred_output:
            boundary_positions_sample = _format_boundary_positions(
                output.bpred_output[0].boundary_mask[0]
            )

        for stage_idx, router_output in enumerate(output.bpred_output):
            while len(selected_sum_by_stage) <= stage_idx:
                selected_sum_by_stage.append(0.0)
                mean_prob_sum_by_stage.append(0.0)
                ratio_gap_sum_by_stage.append(0.0)
                count_by_stage.append(0)

            selected_fraction = float(
                router_output.boundary_mask.float().mean().detach()
            )
            mean_prob = float(
                router_output.boundary_prob[..., -1].float().mean().detach()
            )
            target_ratio = training_config.compression_ratios[
                min(stage_idx, len(training_config.compression_ratios) - 1)
            ]
            target_fraction = 1.0 / max(target_ratio, 1e-8)
            ratio_gap = selected_fraction - target_fraction

            selected_sum_by_stage[stage_idx] += selected_fraction
            mean_prob_sum_by_stage[stage_idx] += mean_prob
            ratio_gap_sum_by_stage[stage_idx] += ratio_gap
            count_by_stage[stage_idx] += 1

    if processed_batches == 0 or token_count == 0:
        return None

    validation_ce = ce_sum / token_count
    validation_bpb = validation_ce / math.log(2.0)

    selected_fraction_by_stage: list[float] = []
    target_ratio_gap_by_stage: list[float] = []
    mean_prob_by_stage: list[float] = []
    for idx, count in enumerate(count_by_stage):
        denom = max(1, count)
        selected_fraction_by_stage.append(selected_sum_by_stage[idx] / denom)
        target_ratio_gap_by_stage.append(ratio_gap_sum_by_stage[idx] / denom)
        mean_prob_by_stage.append(mean_prob_sum_by_stage[idx] / denom)

    if selected_fraction_by_stage:
        actual_selected_fraction = sum(selected_fraction_by_stage) / len(
            selected_fraction_by_stage
        )
        mean_boundary_probability = sum(mean_prob_by_stage) / len(mean_prob_by_stage)
        target_ratio_gap = sum(target_ratio_gap_by_stage) / len(
            target_ratio_gap_by_stage
        )
    else:
        actual_selected_fraction = 0.0
        mean_boundary_probability = 0.0
        target_ratio_gap = 0.0

    stage0_selected_fraction = (
        selected_fraction_by_stage[0]
        if len(selected_fraction_by_stage) > 0
        else float("nan")
    )
    stage1_selected_fraction = (
        selected_fraction_by_stage[1]
        if len(selected_fraction_by_stage) > 1
        else float("nan")
    )
    stage0_target_ratio_gap = (
        target_ratio_gap_by_stage[0]
        if len(target_ratio_gap_by_stage) > 0
        else float("nan")
    )
    stage1_target_ratio_gap = (
        target_ratio_gap_by_stage[1]
        if len(target_ratio_gap_by_stage) > 1
        else float("nan")
    )

    avg_bytes_per_chunk = (
        1.0 / max(stage0_selected_fraction, 1e-8)
        if not math.isnan(stage0_selected_fraction)
        else float("nan")
    )
    compression_l1_l0 = avg_bytes_per_chunk
    compression_l2_l1 = (
        1.0 / max(stage1_selected_fraction, 1e-8)
        if not math.isnan(stage1_selected_fraction)
        else float("nan")
    )

    if not math.isnan(stage0_selected_fraction) and not math.isnan(
        stage1_selected_fraction
    ):
        compression_l2_l0 = 1.0 / max(
            stage0_selected_fraction * stage1_selected_fraction, 1e-8
        )
    else:
        compression_l2_l0 = float("nan")

    return {
        "validation_batches": processed_batches,
        "validation_ce_loss": validation_ce,
        "validation_bpb": validation_bpb,
        "avg_bytes_per_chunk": avg_bytes_per_chunk,
        "target_ratio_gap": target_ratio_gap,
        "actual_selected_fraction": actual_selected_fraction,
        "mean_boundary_probability": mean_boundary_probability,
        "boundary_positions_sample": boundary_positions_sample,
        "stage0_selected_fraction": stage0_selected_fraction,
        "stage1_selected_fraction": stage1_selected_fraction,
        "stage0_target_ratio_gap": stage0_target_ratio_gap,
        "stage1_target_ratio_gap": stage1_target_ratio_gap,
        "compression_l1_l0": compression_l1_l0,
        "compression_l2_l1": compression_l2_l1,
        "compression_l2_l0": compression_l2_l0,
    }


def train(training_config: TrainingConfig) -> None:
    logger = configure_logging()
    set_seed(training_config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)
    logger.info("training_config=%s", asdict(training_config))

    training_dtype = get_training_dtype(device)
    use_grad_scaler = device.type == "cuda" and training_dtype == torch.float16
    logger.info("training_dtype=%s grad_scaler=%s", training_dtype, use_grad_scaler)

    model, model_config = create_model(training_config, device, training_dtype)
    output_dir = Path(training_config.output_dir)
    saved_config_path = save_hnet_config(model_config, output_dir / "model_config.json")
    logger.info("saved_model_config=%s", saved_config_path)
    saved_training_config_path = save_training_config(
        training_config, output_dir / "training_config.json"
    )
    logger.info("saved_training_config=%s", saved_training_config_path)

    metrics_logger = TrainingMetricsLogger(output_dir / "training_metrics.csv")
    logger.info("training_metrics_csv=%s", metrics_logger.output_path)

    validation_metrics_logger = ValidationMetricsLogger(
        output_dir / "validation_metrics.csv"
    )
    logger.info("validation_metrics_csv=%s", validation_metrics_logger.output_path)

    train_sources = training_config.datasets
    validation_sources: list[DatasetSource]

    if training_config.validation_datasets is not None:
        validation_sources = training_config.validation_datasets
        logger.info("validation_source_mode=explicit")
    else:
        train_sources, validation_sources = split_train_validation_sources(
            training_config.datasets,
            training_config.validation_split_ratio,
        )
        logger.info(
            "validation_source_mode=split_from_train ratio=%.4f (fixed holdout)",
            training_config.validation_split_ratio,
        )

    estimated_examples, estimated_optimizer_steps = estimate_dataset_examples(
        training_config,
        sources=train_sources,
    )
    if estimated_examples is None:
        logger.info("dataset_examples_estimate=unavailable")
    else:
        logger.info(
            "dataset_examples_estimate=%d data_mix=concatenate_then_shuffle rough_optimizer_steps=%d",
            estimated_examples,
            estimated_optimizer_steps,
        )

    target_steps = training_config.max_steps
    if target_steps is None:
        logger.info("epoch_count_mode=streaming_no_prescan")
        if estimated_optimizer_steps is not None:
            logger.info(
                "target_optimizer_steps_estimate=%d (used for approximate progress and lr decay)",
                estimated_optimizer_steps,
            )
        else:
            logger.info("target_optimizer_steps_estimate=unavailable")
    else:
        logger.info("target_optimizer_steps=%s", target_steps)

    lr_total_steps = target_steps or estimated_optimizer_steps
    if lr_total_steps is None:
        logger.info("lr_schedule=warmup_then_constant (no total_steps estimate)")
    else:
        logger.info(
            "lr_schedule=wsd warmup=10%% stable=70%% decay=20%% decay_shape=inverse_sqrt"
        )

    total_params, trainable_params = count_parameters(model)
    logger.info(
        "model_parameters total=%s trainable=%s",
        format_parameter_count(total_params),
        format_parameter_count(trainable_params),
    )

    dataloader = create_dataloader(
        training_config,
        sources=train_sources,
        shuffle=True,
    )
    data_iterator = iter(dataloader)

    param_groups = group_params(model)
    apply_weight_decay(param_groups, training_config.weight_decay)
    optimizer = AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    validation_dataloader = create_dataloader(
        training_config,
        sources=validation_sources,
        shuffle=False,
    )
    validation_batches = build_cached_validation_batches(
        validation_dataloader,
        training_config.validation_max_batches,
    )
    logger.info("cached_validation_batches=%d", len(validation_batches))

    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    last_saved_step = 0
    completed_steps = 0

    while (
        training_config.max_steps is None
        or completed_steps < training_config.max_steps
    ):
        optimizer.zero_grad(set_to_none=True)
        ce_loss_value = 0.0
        ratio_loss_value = 0.0
        micro_batches_processed = 0

        if lr_total_steps is None:
            if completed_steps < training_config.warmup_steps:
                learning_rate = (
                    training_config.learning_rate
                    * float(completed_steps + 1)
                    / float(max(1, training_config.warmup_steps))
                )
            else:
                learning_rate = training_config.learning_rate
        else:
            learning_rate = wsd_schedule(
                step=completed_steps,
                total_steps=lr_total_steps,
                max_lr=training_config.learning_rate,
                min_lr=training_config.min_learning_rate,
            )
        apply_learning_rate(optimizer, learning_rate)

        for _micro_step in range(training_config.grad_accum_steps):
            try:
                batch = next(data_iterator)
            except StopIteration:
                break
            micro_batches_processed += 1
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=training_dtype,
                enabled=device.type == "cuda",
            ):
                output = model(input_ids=input_ids, mask=mask)
                ce_loss = F.cross_entropy(
                    output.logits.reshape(-1, output.logits.shape[-1]),
                    labels.reshape(-1),
                )
                ratio_loss = compute_ratio_loss(
                    output.bpred_output,
                    training_config.compression_ratios,
                    device,
                ).to(dtype=ce_loss.dtype)
                loss = ce_loss + training_config.train_ratio_weight * ratio_loss
                loss = loss / training_config.grad_accum_steps

            if use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            ce_loss_value += float(ce_loss.detach())
            ratio_loss_value += float(ratio_loss.detach())

        if micro_batches_processed == 0:
            if training_config.max_steps is None:
                logger.info("epoch_completed=true completed_steps=%d", completed_steps)
            else:
                logger.warning(
                    "data_exhausted_before_target=true completed_steps=%d target_steps=%d",
                    completed_steps,
                    training_config.max_steps,
                )
            break

        if use_grad_scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), training_config.grad_clip_norm
        )
        if use_grad_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        completed_steps += 1
        average_ce = ce_loss_value / micro_batches_processed
        average_ratio = ratio_loss_value / micro_batches_processed
        total_loss = average_ce + training_config.train_ratio_weight * average_ratio
        metrics_logger.log(
            step=completed_steps,
            learning_rate=learning_rate,
            ce_loss=average_ce,
            ratio_loss=average_ratio,
            total_loss=total_loss,
        )

        if completed_steps % training_config.log_every == 0 or completed_steps == 1:
            if training_config.max_steps is not None:
                epoch_progress = (
                    100.0 * completed_steps / float(max(1, training_config.max_steps))
                )
                logger.info(
                    "step=%d/%d epoch_progress=%.2f%% lr=%.6g ce_loss=%.4f ratio_loss=%.4f total_loss=%.4f",
                    completed_steps,
                    training_config.max_steps,
                    epoch_progress,
                    learning_rate,
                    average_ce,
                    average_ratio,
                    total_loss,
                )
            elif estimated_optimizer_steps is not None:
                epoch_progress = (
                    100.0 * completed_steps / float(max(1, estimated_optimizer_steps))
                )
                logger.info(
                    "step=%d epoch_progress_est=%.2f%% lr=%.6g ce_loss=%.4f ratio_loss=%.4f total_loss=%.4f",
                    completed_steps,
                    epoch_progress,
                    learning_rate,
                    average_ce,
                    average_ratio,
                    total_loss,
                )
            else:
                logger.info(
                    "step=%d epoch_progress=unavailable lr=%.6g ce_loss=%.4f ratio_loss=%.4f total_loss=%.4f",
                    completed_steps,
                    learning_rate,
                    average_ce,
                    average_ratio,
                    total_loss,
                )

        if (
            training_config.validation_every > 0
            and completed_steps % training_config.validation_every == 0
        ):
            model.eval()
            validation_metrics = evaluate_validation(
                model=model,
                training_config=training_config,
                device=device,
                training_dtype=training_dtype,
                validation_batches=validation_batches,
            )
            model.train()

            if validation_metrics is not None:
                validation_metrics_logger.log(
                    step=completed_steps,
                    metrics=validation_metrics,
                )
                logger.info(
                    "validation step=%d ce=%.4f bpb=%.4f comp(L1/L0)=%.3f comp(L2/L1)=%.3f comp(L2/L0)=%.3f sf0=%.4f sf1=%.4f gap0=%.4f gap1=%.4f",
                    completed_steps,
                    validation_metrics["validation_ce_loss"],
                    validation_metrics["validation_bpb"],
                    validation_metrics["compression_l1_l0"],
                    validation_metrics["compression_l2_l1"],
                    validation_metrics["compression_l2_l0"],
                    validation_metrics["stage0_selected_fraction"],
                    validation_metrics["stage1_selected_fraction"],
                    validation_metrics["stage0_target_ratio_gap"],
                    validation_metrics["stage1_target_ratio_gap"],
                )

        if completed_steps % training_config.save_every == 0:
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=completed_steps,
                output_dir=output_dir,
            )
            last_saved_step = completed_steps
            logger.info("saved_checkpoint=%s", checkpoint_path)

    if completed_steps > 0 and last_saved_step != completed_steps:
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=completed_steps,
            output_dir=output_dir,
        )
        logger.info("saved_final_checkpoint=%s", checkpoint_path)
