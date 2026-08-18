from argparse import Namespace
from pathlib import Path

from scripts.run_phase2_screening import build_command


def _args(model: str) -> Namespace:
    return Namespace(
        model=model,
        seed=42,
        packed_data_dir=Path("train"),
        packed_validation_data_dir=Path("validation"),
        max_train_bytes=500_000_000,
        checkpoint_every_bytes=125_000_000,
        batch_size=4,
        grad_accum_steps=32,
        learning_rate=3e-4,
        min_learning_rate=3e-5,
        lr_schedule_steps=2000,
    )


def test_hnet_phase2_command_uses_raw_byte_budget() -> None:
    command = build_command(_args("t26"), Path("output"))

    assert command[command.index("--max-train-bytes") + 1] == "500000000"
    assert command[command.index("--save-every-bytes") + 1] == "125000000"
    assert command[command.index("--byte-boundary-constraint") + 1] == "utf8-hard"
    assert command.count("--compression-ratio") == 2


def test_tokenizer_phase2_command_disables_byte_boundary_rules() -> None:
    command = build_command(_args("tokenizer"), Path("output"))

    assert command[command.index("--byte-boundary-constraint") + 1] == "off"
    assert command[command.index("--train-ratio-weight") + 1] == "0.0"
    assert command.count("--compression-ratio") == 1
