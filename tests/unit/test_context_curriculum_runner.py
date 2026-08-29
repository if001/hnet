from argparse import Namespace
from pathlib import Path

from scripts.run_context_curriculum_training import (
    BASE_SEQ_LEN,
    BYTES_PER_UPDATE,
    LEGS,
    REQUIRED_CANONICAL_BLOCKS,
    TOTAL_TRAIN_BYTES,
    build_train_command,
    expected_validation_steps,
)


def make_args() -> Namespace:
    return Namespace(
        main="t26",
        seed=42,
        model_init_seed=42,
        data_order_seed=42,
        train_runtime_seed=42,
        packed_data_dir=Path("train-packed"),
        packed_validation_data_dir=Path("validation-packed"),
        initial_model_checkpoint=None,
        save_initial_model_to=None,
    )


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_curriculum_legs_keep_one_canonical_block_per_micro_batch() -> None:
    assert REQUIRED_CANONICAL_BLOCKS == 600 * 32
    for name in ("shared-a", "l1-b", "l1-c"):
        leg = LEGS[name]
        assert leg.seq_len * leg.batch_size == BASE_SEQ_LEN
        assert leg.seq_len * leg.batch_size * 32 == BYTES_PER_UPDATE


def test_l1_b_command_uses_global_byte_schedule_and_transition_evaluations() -> None:
    args = make_args()
    leg = LEGS["l1-b"]
    command = build_train_command(
        args,
        leg,
        Path("run"),
        Path("phase-a/checkpoint_step_000200.pt"),
        ["probe"],
    )

    assert option_value(command, "--seq-len") == "8192"
    assert option_value(command, "--batch-size") == "4"
    assert option_value(command, "--max-steps") == "400"
    assert option_value(command, "--max-train-bytes") == str(TOTAL_TRAIN_BYTES)
    assert option_value(command, "--lr-schedule-steps") == "600"
    assert option_value(command, "--packed-curriculum-base-seq-len") == "32768"
    transition_indices = [
        index
        for index, value in enumerate(command)
        if value == "--validation-step"
    ]
    assert [command[index + 1] for index in transition_indices] == ["201", "205"]
    assert option_value(command, "--resume-from-checkpoint").endswith(
        "checkpoint_step_000200.pt"
    )


def test_expected_validation_steps_include_transition_drawdown_points() -> None:
    steps = expected_validation_steps(LEGS["l1-c"])
    assert steps[:4] == (401, 405, 410, 420)
    assert steps[-1] == 600
