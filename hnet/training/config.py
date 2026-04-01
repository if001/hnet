from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSource:
    name: str
    split: str = "train"
    config_name: str | None = None
    take_examples: int = -1
    skip_examples: int = 0


@dataclass(frozen=True)
class TrainingConfig:
    model_config_path: str
    output_dir: str = "artifacts"
    datasets: list[DatasetSource] = field(
        default_factory=lambda: [
            DatasetSource(name="if001/bunpo_phi4_ctx"),
            DatasetSource(name="if001/bunpo_phi4"),
        ]
    )
    seq_len: int = 512
    batch_size: int = 2
    grad_accum_steps: int = 8
    max_steps: int | None = None
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 20
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    log_every: int = 10
    save_every: int = 100
    train_ratio_weight: float = 0.02
    compression_ratios: list[float] = field(default_factory=lambda: [4.0])
    lr_multipliers: list[float] = field(default_factory=lambda: [1.0, 1.0])
    seed: int = 42
    num_workers: int = 0
    shuffle_buffer_size: int = 512
