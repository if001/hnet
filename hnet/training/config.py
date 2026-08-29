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
    packed_data_dir: str | None = None
    packed_validation_data_dir: str | None = None
    packed_curriculum_base_seq_len: int | None = None
    packed_curriculum_group_weights: dict[str, float] | None = None
    validation_datasets: list[DatasetSource] | None = None
    chunk_prompts: list[str] = field(default_factory=list)
    seq_len: int = 512
    batch_size: int = 2
    grad_accum_steps: int = 8
    max_steps: int | None = None
    max_train_bytes: int | None = None
    lr_schedule_steps: int | None = None
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 20
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    log_every: int = 10
    save_every: int = 100
    save_every_bytes: int | None = None
    validation_every: int = 100
    validation_every_bytes: int | None = None
    validation_steps: list[int] = field(default_factory=list)
    validation_max_batches: int = 20
    validation_split_ratio: float = 0.1
    train_ratio_weight: float = 0.02
    byte_boundary_constraint: str = "off"
    byte_boundary_constraint_weight: float = 0.0
    byte_boundary_constraint_bias: float = 0.0
    family_consistency_data: str | None = None
    family_consistency_split: str = "train"
    family_consistency_objective: str = "c1"
    family_consistency_weight: float = 0.0
    family_consistency_margin: float = 0.15
    family_consistency_seed: int = 42
    moe_aux_loss_weight: float = 0.0
    compression_ratios: list[float] = field(default_factory=lambda: [4.0])
    lr_multipliers: list[float] = field(default_factory=lambda: [1.0, 1.0])
    seed: int = 42
    model_init_seed: int | None = None
    data_order_seed: int | None = None
    train_runtime_seed: int | None = None
    num_workers: int = 0
    shuffle_buffer_size: int = 512
    initial_model_checkpoint: str | None = None
    save_initial_model_to: str | None = None
    resume_from_checkpoint: str | None = None
    resume_optimizer: bool = True
    resume_step: bool = True
    freeze_mode: str = "none"
    rope_scaling: dict[str, float | int | str] | None = None
