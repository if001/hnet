from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    Trainer,
    TrainingArguments,
)

from hnet.models import HNetForCausalLM
from hnet.utils.train import group_params, load_balancing_loss

from .data import StreamingSFTByteDataset


@dataclass(frozen=True)
class SFTTrainConfig:
    model_config_path: str
    pretrained_model_path: str
    output_dir: str = "artifacts/hnet_sft"
    chat_tokenizer_path: str = "Qwen/Qwen3-0.6B"
    mix_config_path: str | None = None

    seq_len: int = 512
    packing: bool = True
    batch_size: int = 2
    grad_accum_steps: int = 8
    max_steps: int | None = None

    learning_rate: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0

    train_ratio_weight: float = 0.02
    compression_ratios: list[float] | None = None
    lr_multipliers: list[float] | None = None

    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 42
    num_workers: int = 0
    shuffle_buffer_size: int = 512


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


class HNetSFTTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        ratio_weight: float,
        compression_ratios: list[float],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ratio_weight = ratio_weight
        self.compression_ratios = compression_ratios

    def compute_loss(
        self,
        model: HNetForCausalLM,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        del num_items_in_batch
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        mask = inputs.get("mask")

        outputs = model(input_ids=input_ids, mask=mask)
        vocab_size = outputs.logits.shape[-1]
        ce_loss = F.cross_entropy(
            outputs.logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        ratio_loss = compute_ratio_loss(
            outputs.bpred_output,
            self.compression_ratios,
            device=ce_loss.device,
        )
        total_loss = ce_loss + self.ratio_weight * ratio_loss

        # if self.model.training:
        #     self.log(
        #         {
        #             "ce_loss": float(ce_loss.detach().cpu()),
        #             "ratio_loss": float(ratio_loss.detach().cpu()),
        #             "total_loss": float(total_loss.detach().cpu()),
        #         }
        #     )

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def create_optimizer(self) -> AdamW:
        if self.optimizer is None:
            model = self.model
            if not isinstance(model, HNetForCausalLM):
                raise TypeError("HNetSFTTrainer expects HNetForCausalLM model")

            parameter_groups = group_params(model)
            for group in parameter_groups:
                if "weight_decay" not in group:
                    group["weight_decay"] = self.args.weight_decay

                multiplier = float(group.get("lr_multiplier", 1.0))
                group["lr"] = self.args.learning_rate * multiplier

            self.optimizer = AdamW(
                parameter_groups,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

        return self.optimizer


def build_training_arguments(
    config: SFTTrainConfig, *, max_steps: int | None = None
) -> TrainingArguments:
    effective_max_steps = config.max_steps if max_steps is None else max_steps
    if effective_max_steps is None:
        raise ValueError("max_steps must be set before building TrainingArguments")

    return TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        max_steps=effective_max_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.grad_clip_norm,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_strategy="steps",
        save_total_limit=3,
        dataloader_num_workers=config.num_workers,
        dataloader_persistent_workers=config.num_workers > 0,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        # report_to=[],
        report_to="wandb",
        seed=config.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
    )
