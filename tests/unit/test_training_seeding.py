import random

import numpy as np
import torch

from hnet.training.config import TrainingConfig
from hnet.training.seeding import (
    capture_rng_state,
    model_state_sha256,
    resolve_training_seeds,
    restore_rng_state,
    set_global_seed,
)


def test_split_seeds_fall_back_to_legacy_seed() -> None:
    config = TrainingConfig(model_config_path="model.json", seed=17)
    resolved = resolve_training_seeds(config)
    assert resolved.model_init_seed == 17
    assert resolved.data_order_seed == 17
    assert resolved.train_runtime_seed == 17


def test_split_seeds_override_independently() -> None:
    config = TrainingConfig(
        model_config_path="model.json",
        seed=17,
        model_init_seed=18,
        data_order_seed=19,
        train_runtime_seed=20,
    )
    resolved = resolve_training_seeds(config)
    assert resolved.model_init_seed == 18
    assert resolved.data_order_seed == 19
    assert resolved.train_runtime_seed == 20


def test_rng_state_round_trip_restores_all_cpu_generators() -> None:
    set_global_seed(23)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    set_global_seed(99)
    restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_restore_rng_state_normalizes_cuda_states_to_cpu_byte_tensors(
    monkeypatch,
) -> None:
    set_global_seed(23)
    state = capture_rng_state()
    state["torch_cuda"] = [torch.arange(8, dtype=torch.int64)]
    restored: list[list[torch.Tensor]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.append)

    restore_rng_state(state)

    assert len(restored) == 1
    assert len(restored[0]) == 1
    assert restored[0][0].device.type == "cpu"
    assert restored[0][0].dtype == torch.uint8
    assert restored[0][0].is_contiguous()


def test_model_state_hash_tracks_exact_initial_values() -> None:
    set_global_seed(31)
    left = torch.nn.Linear(4, 3)
    set_global_seed(31)
    right = torch.nn.Linear(4, 3)
    assert model_state_sha256(left) == model_state_sha256(right)
    with torch.no_grad():
        right.weight[0, 0] += 1
    assert model_state_sha256(left) != model_state_sha256(right)
