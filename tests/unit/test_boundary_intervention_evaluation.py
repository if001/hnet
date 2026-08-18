import math

import pytest
import torch

from scripts.evaluate_boundary_interventions import bits_per_raw_byte


def test_bits_per_raw_byte_uses_raw_byte_denominator() -> None:
    token_loss = torch.tensor([1.0, 2.0, 3.0])

    assert bits_per_raw_byte(token_loss, raw_bytes=2) == pytest.approx(
        3.0 / math.log(2.0)
    )


def test_bits_per_raw_byte_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="raw_bytes must be positive"):
        bits_per_raw_byte(torch.tensor([1.0]), raw_bytes=0)
