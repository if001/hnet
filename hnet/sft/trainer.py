from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping

import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset
from torch.optim import AdamW
from torch.utils.data import IterableDataset
from transformers import Trainer, TrainingArguments

from hnet.models import HNetForCausalLM
from hnet.training.config import DatasetSource
from hnet.training.data import DefaultRecordFormatter
from hnet.utils.tokenizers import ByteTokenizer
from hnet.utils.train import group_params, load_balancing_loss

from .dataset import SFTDataConfig, build_sft_train_dataset


@dataclass(frozen=True)
class SFTTrainConfig:
    model_config_path: str
    pretrained_model_path: str
    output_dir: str = "artifacts/hnet_sft"

    datasets: list[DatasetSource] | None = None
    use_sample_dataset: bool = True

    seq_len: int = 512
    batch_size: int = 2
    grad_accum_steps: int = 8
    max_steps: int = 1000

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


def _iter_custom_dataset_records(
    sources: list[DatasetSource],
    shuffle_buffer_size: int,
    seed: int,
) -> Iterable[Mapping[str, object]]:
    datasets = []
    for source in sources:
        dataset = load_dataset(
            source.name,
            source.config_name,
            split=source.split,
            streaming=True,
        )
        if source.skip_examples > 0:
            dataset = dataset.skip(source.skip_examples)
        if source.take_examples > 0:
            dataset = dataset.take(source.take_examples)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
        datasets.append(dataset)

    if not datasets:
        return []
    if len(datasets) == 1:
        return datasets[0]

    merged = concatenate_datasets(datasets)
    return merged.shuffle(buffer_size=shuffle_buffer_size, seed=seed)


class StreamingSFTByteDataset(IterableDataset):
    def __init__(
        self,
        *,
        seq_len: int,
        shuffle_buffer_size: int,
        use_sample_dataset: bool,
        sources: list[DatasetSource] | None,
        seed: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.shuffle_buffer_size = shuffle_buffer_size
        self.use_sample_dataset = use_sample_dataset
        self.sources = sources
        self.seed = seed
        self.tokenizer = ByteTokenizer()
        self.formatter = DefaultRecordFormatter()

    def _iter_texts(self) -> Iterator[str]:
        if self.use_sample_dataset and not self.sources:
            sample_cfg = SFTDataConfig(
                seed=self.seed,
                shuffle_buffer_size=self.shuffle_buffer_size,
            )
            dataset = build_sft_train_dataset(sample_cfg)
            for example in dataset:
                text = self.formatter.format_record(example)
                if text:
                    yield text
            return

        if not self.sources:
            raise ValueError(
                "Either --dataset must be provided or sample dataset mode must be enabled"
            )

        for record in _iter_custom_dataset_records(
            self.sources,
            shuffle_buffer_size=self.shuffle_buffer_size,
            seed=self.seed,
        ):
            text = self.formatter.format_record(record)
            if text:
                yield text

    def __iter__(self):
        token_buffer: list[int] = []

        for text in self._iter_texts():
            encoded = self.tokenizer.encode([text], add_bos=True, add_eos=True)[0][
                "input_ids"
            ].tolist()
            token_buffer.extend(encoded)

            while len(token_buffer) >= self.seq_len + 1:
                chunk = token_buffer[: self.seq_len + 1]
                del token_buffer[: self.seq_len + 1]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                mask = torch.ones(self.seq_len, dtype=torch.bool)
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "mask": mask,
                }


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


def build_training_arguments(config: SFTTrainConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.grad_clip_norm,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_strategy="steps",
        save_total_limit=3,
        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        # report_to=[],
        report_to="wandb",
        seed=config.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
    )
