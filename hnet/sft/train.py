from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch
from omegaconf import ListConfig
from transformers import set_seed

from hnet.models import HNetForCausalLM, load_hnet_config
from hnet.models.config_io import save_hnet_config

from hnet.sft.trainer import (
    HNetSFTTrainer,
    SFTTrainConfig,
    StreamingSFTByteDataset,
    build_training_arguments,
)


def extract_model_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        model_state = checkpoint["model"]
        if isinstance(model_state, Mapping):
            return model_state
    if isinstance(checkpoint, Mapping):
        return checkpoint
    raise TypeError("Unsupported checkpoint format")


def load_pretrained_state(model_path: str) -> Mapping[str, torch.Tensor]:
    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 6):
        with torch.serialization.safe_globals([ListConfig]):
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    else:
        checkpoint = torch.load(model_path, map_location="cpu")
    return extract_model_state_dict(checkpoint)


def parse_args() -> SFTTrainConfig:
    parser = argparse.ArgumentParser(
        description="SFT training for H-Net with Hugging Face Trainer"
    )
    parser.add_argument("--model-config-path", type=str, required=True)
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="artifacts/hnet_sft")
    parser.add_argument(
        "--chat-tokenizer-path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Tokenizer used for Qwen3 chat template rendering.",
    )

    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

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

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle-buffer-size", type=int, default=512)

    args = parser.parse_args()

    compression_ratios = args.compression_ratios or [4.0]
    lr_multipliers = args.lr_multipliers or [1.0, 1.0]

    return SFTTrainConfig(
        model_config_path=args.model_config_path,
        pretrained_model_path=args.pretrained_model_path,
        output_dir=args.output_dir,
        chat_tokenizer_path=args.chat_tokenizer_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        train_ratio_weight=args.train_ratio_weight,
        compression_ratios=compression_ratios,
        lr_multipliers=lr_multipliers,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )


def train(config: SFTTrainConfig) -> None:
    set_seed(config.seed)

    hnet_config = load_hnet_config(config.model_config_path)
    model = HNetForCausalLM(hnet_config)
    model.apply_lr_multiplier(config.lr_multipliers or [1.0, 1.0])

    state_dict = load_pretrained_state(config.pretrained_model_path)
    model.load_state_dict(state_dict, strict=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model_parameters_total={total_params}")
    print(f"model_parameters_trainable={trainable_params}")

    train_dataset = StreamingSFTByteDataset(
        seq_len=config.seq_len,
        shuffle_buffer_size=config.shuffle_buffer_size,
        seed=config.seed,
        chat_tokenizer_path=config.chat_tokenizer_path,
    )

    training_args = build_training_arguments(config)
    trainer = HNetSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        ratio_weight=config.train_ratio_weight,
        compression_ratios=config.compression_ratios or [4.0],
    )

    trainer.train()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_hnet_config(hnet_config, output_dir / "model_config.json")
    (output_dir / "sft_training_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    final_checkpoint = {
        "model": trainer.model.state_dict(),
        "step": int(trainer.state.global_step),
    }
    torch.save(final_checkpoint, output_dir / "sft_final_model.pt")
    print(f"saved_final_checkpoint={output_dir / 'sft_final_model.pt'}")


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
