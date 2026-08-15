from hnet.training.config import TrainingConfig


def test_lr_schedule_horizon_defaults_to_unspecified() -> None:
    config = TrainingConfig(model_config_path="model.json")

    assert config.lr_schedule_steps is None


def test_lr_schedule_horizon_can_exceed_training_steps() -> None:
    config = TrainingConfig(
        model_config_path="model.json", max_steps=55, lr_schedule_steps=220
    )

    assert config.max_steps == 55
    assert config.lr_schedule_steps == 220
