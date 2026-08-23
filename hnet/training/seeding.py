from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .config import TrainingConfig


@dataclass(frozen=True)
class ResolvedTrainingSeeds:
    model_init_seed: int
    data_order_seed: int
    train_runtime_seed: int


def resolve_training_seeds(config: TrainingConfig) -> ResolvedTrainingSeeds:
    """Resolve split seeds while preserving the historical single-seed behavior."""
    return ResolvedTrainingSeeds(
        model_init_seed=(
            config.seed if config.model_init_seed is None else config.model_init_seed
        ),
        data_order_seed=(
            config.seed if config.data_order_seed is None else config.data_order_seed
        ),
        train_runtime_seed=(
            config.seed
            if config.train_runtime_seed is None
            else config.train_runtime_seed
        ),
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_cpu_state = state.get("torch_cpu")
    if python_state is None or numpy_state is None or not torch.is_tensor(torch_cpu_state):
        raise ValueError("checkpoint RNG state is incomplete")
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_cpu_state.cpu())
    cuda_state = state.get("torch_cuda")
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def model_state_sha256(model: torch.nn.Module) -> str:
    """Hash exact state tensor bytes one tensor at a time to limit peak CPU memory."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().contiguous()
        raw = value.reshape(-1).view(torch.uint8).cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(raw.tobytes())
    return digest.hexdigest()
