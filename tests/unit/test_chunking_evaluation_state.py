import random

import numpy as np
import torch

from hnet.training.chunking_utils import deterministic_model_evaluation


def test_deterministic_evaluation_restores_mode_and_rng_states() -> None:
    model = torch.nn.Dropout(p=0.5)
    model.train()
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)

    with deterministic_model_evaluation(model):
        assert model.training is False
        random.random()
        np.random.rand()
        torch.rand(())

    assert model.training is True
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected


def test_deterministic_evaluation_preserves_eval_mode() -> None:
    model = torch.nn.Linear(2, 2)
    model.eval()
    with deterministic_model_evaluation(model):
        assert model.training is False
    assert model.training is False
