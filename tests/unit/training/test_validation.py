import csv
import json
import math
from types import SimpleNamespace

import torch

from hnet.training.config import TrainingConfig
from hnet.training.trainer import ValidationMetricsLogger, evaluate_validation
from hnet.training.validation import load_saved_training_config


def test_load_saved_training_config_restores_dataset_sources(tmp_path) -> None:
    path = tmp_path / "training_config.json"
    path.write_text(
        json.dumps(
            {
                "model_config_path": "model.json",
                "datasets": [
                    {
                        "name": "example/train",
                        "split": "train",
                        "take_examples": 10,
                        "unknown_future_field": True,
                    }
                ],
                "validation_datasets": [
                    {"name": "example/validation", "split": "test"}
                ],
                "seq_len": 2048,
                "unknown_config_field": "ignored",
            }
        ),
        encoding="utf-8",
    )

    config = load_saved_training_config(path)

    assert config.seq_len == 2048
    assert config.datasets[0].name == "example/train"
    assert config.datasets[0].take_examples == 10
    assert config.validation_datasets is not None
    assert config.validation_datasets[0].name == "example/validation"
    assert config.validation_datasets[0].split == "test"


def test_validation_bpb_is_normalized_by_raw_bytes() -> None:
    class UniformModel(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            del kwargs
            logits = torch.zeros((*input_ids.shape, 2), device=input_ids.device)
            return SimpleNamespace(logits=logits, bpred_output=[])

    config = TrainingConfig(
        model_config_path="unused.json",
        validation_max_batches=1,
    )
    batch = {
        "input_ids": torch.tensor([[0, 1]]),
        "labels": torch.tensor([[0, 1]]),
        "mask": torch.ones((1, 2), dtype=torch.bool),
        "target_byte_lengths": torch.tensor([[1, 3]]),
        "is_byte_level": torch.tensor([False]),
    }

    metrics = evaluate_validation(
        UniformModel(),
        config,
        torch.device("cpu"),
        torch.float32,
        [batch],
    )

    assert metrics is not None
    assert math.isclose(
        metrics["validation_ce_loss"], math.log(2.0), rel_tol=1e-6
    )
    assert math.isclose(metrics["validation_bpb"], 0.5, rel_tol=1e-6)


def test_validation_metrics_logger_migrates_old_header_on_resume(tmp_path) -> None:
    path = tmp_path / "validation_metrics.csv"
    path.write_text(
        "step,validation_batches,validation_ce_loss\n4,2,1.5\n",
        encoding="utf-8",
    )

    ValidationMetricsLogger(path, append=True)

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["step"] == "4"
    assert rows[0]["cumulative_input_bytes"] == ""
