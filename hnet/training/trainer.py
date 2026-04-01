import csv
import json
import logging
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..models.config_hnet import HNetConfig
from ..models.config_io import load_hnet_config, save_hnet_config
from ..models.mixer_seq import HNetForCausalLM
from ..utils.train import group_params, load_balancing_loss
from .config import TrainingConfig
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
) -> DataLoader[dict[str, torch.Tensor]]:
    dataset = StreamingByteDataset(
        sources=training_config.datasets,
        seq_len=training_config.seq_len,
        formatter=DefaultRecordFormatter(),
        shuffle_buffer_size=training_config.shuffle_buffer_size,
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


def estimate_dataset_examples(training_config: TrainingConfig) -> tuple[int | None, int | None]:
    example_counts = []
    for source in training_config.datasets:
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


def count_epoch_micro_batches(training_config: TrainingConfig) -> int:
    dataloader = create_dataloader(training_config)
    return sum(1 for _ in dataloader)


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

    estimated_examples, estimated_optimizer_steps = estimate_dataset_examples(training_config)
    if estimated_examples is None:
        logger.info("dataset_examples_estimate=unavailable")
    else:
        logger.info(
            "dataset_examples_estimate=%d data_mix=concatenate_then_shuffle rough_optimizer_steps=%d",
            estimated_examples,
            estimated_optimizer_steps,
        )

    epoch_micro_batches: int | None = None
    if training_config.max_steps is None:
        logger.info("counting_epoch_micro_batches=true")
        epoch_micro_batches = count_epoch_micro_batches(training_config)
        logger.info("epoch_micro_batches=%d", epoch_micro_batches)

    if training_config.max_steps is None:
        if epoch_micro_batches is None:
            raise RuntimeError("failed to compute epoch_micro_batches")
        target_steps = max(
            1, math.ceil(epoch_micro_batches / training_config.grad_accum_steps)
        )
    else:
        target_steps = training_config.max_steps
    logger.info("target_optimizer_steps=%s", target_steps)
    logger.info("lr_schedule=wsd warmup=10%% stable=70%% decay=20%% decay_shape=inverse_sqrt")

    total_params, trainable_params = count_parameters(model)
    logger.info(
        "model_parameters total=%s trainable=%s",
        format_parameter_count(total_params),
        format_parameter_count(trainable_params),
    )

    dataloader = create_dataloader(training_config)
    data_iterator = iter(dataloader)

    param_groups = group_params(model)
    apply_weight_decay(param_groups, training_config.weight_decay)
    optimizer = AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    last_saved_step = 0
    completed_steps = 0

    while completed_steps < target_steps:
        optimizer.zero_grad(set_to_none=True)
        ce_loss_value = 0.0
        ratio_loss_value = 0.0
        micro_batches_processed = 0

        learning_rate = wsd_schedule(
            step=completed_steps,
            total_steps=target_steps,
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
            logger.warning(
                "data_exhausted_before_target=true completed_steps=%d target_steps=%d",
                completed_steps,
                target_steps,
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
            epoch_progress = 100.0 * completed_steps / float(max(1, target_steps))
            logger.info(
                "step=%d/%d epoch_progress=%.2f%% lr=%.6g ce_loss=%.4f ratio_loss=%.4f total_loss=%.4f",
                completed_steps,
                target_steps,
                epoch_progress,
                learning_rate,
                average_ce,
                average_ratio,
                total_loss,
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
