from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def seed_identity(run: Mapping[str, Any]) -> str:
    factors = run.get("seed_factors")
    if isinstance(factors, Mapping):
        init_seed = factors.get("model_init_seed", run.get("seed"))
        data_seed = factors.get("data_order_seed", run.get("seed"))
        runtime_seed = factors.get("train_runtime_seed", run.get("seed"))
        return f"i{init_seed}_d{data_seed}_r{runtime_seed}"
    return f"s{run.get('seed')}"


def linguistic_run_id(run: Mapping[str, Any]) -> str:
    return (
        f"{run['model_name']}|{seed_identity(run)}|"
        f"{run.get('checkpoint_label')}"
    )
