from pathlib import Path

import torch

from scripts.extract_preconditioned_model import build_preconditioned_payload


def test_build_preconditioned_payload_drops_training_state() -> None:
    model = {"weight": torch.tensor([1.0, 2.0])}
    payload = {
        "model": model,
        "optimizer": {"state": {1: "discard"}},
        "step": 20,
        "data_state": {"micro_batches_seen": 640},
        "rng_state": {"python": "discard"},
    }

    result = build_preconditioned_payload(
        payload,
        source_path=Path("warmup.pt"),
        source_sha256="abc123",
    )

    assert result["model"] is model
    assert result["step"] == 0
    assert result["preconditioning"] == {
        "source_path": "warmup.pt",
        "source_sha256": "abc123",
        "source_step": 20,
    }
    assert "optimizer" not in result
    assert "data_state" not in result
    assert "rng_state" not in result
