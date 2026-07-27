import json

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
