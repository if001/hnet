from types import SimpleNamespace

import torch

from hnet.sft.trainer import HNetSFTTrainer


def test_trainer_checkpoint_preserves_tied_weights(tmp_path):
    shared_weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32))
    model = torch.nn.Module()
    model.register_parameter("embedding_weight", shared_weight)
    model.register_parameter("lm_head_weight", shared_weight)
    trainer = object.__new__(HNetSFTTrainer)
    trainer.model = model
    trainer.args = SimpleNamespace(output_dir=str(tmp_path))

    trainer._save()

    state = torch.load(tmp_path / "pytorch_model.bin", weights_only=True)
    assert state["embedding_weight"].data_ptr() == state["lm_head_weight"].data_ptr()
    assert (tmp_path / "training_args.bin").is_file()
