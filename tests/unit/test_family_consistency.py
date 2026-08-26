from __future__ import annotations

import json

import pytest
import torch

from hnet.modules.dc import RoutingModuleOutput
from hnet.training.family_consistency import (
    FamilyConsistencyPair,
    build_family_pair_batch,
    landmark_consistency_loss,
    load_family_consistency_pairs,
)


def test_family_pair_batch_maps_byte_offset_after_bos() -> None:
    pair = FamilyConsistencyPair(
        "p1", "inflection", "猫が走る。", "犬も走った。", 9, 9
    )
    batch = build_family_pair_batch(pair, device=torch.device("cpu"))
    assert batch.input_ids[:, 0].tolist() == [0xFE, 0xFE]
    assert batch.landmark_positions.tolist() == [10, 10]
    assert batch.mask.shape == batch.input_ids.shape


def test_landmark_consistency_loss_has_gradient() -> None:
    logits = torch.zeros((2, 8), requires_grad=True)
    probability = torch.stack((1 - logits.sigmoid(), logits.sigmoid()), dim=-1)
    output = RoutingModuleOutput(
        boundary_prob=probability,
        boundary_mask=torch.zeros((2, 8), dtype=torch.bool),
        selected_probs=torch.ones((2, 8, 1)),
        valid_mask=torch.ones((2, 8), dtype=torch.bool),
    )
    with torch.no_grad():
        logits[1, 4] = 2.0
    loss = landmark_consistency_loss(output, torch.tensor([3, 4]))
    loss.backward()
    assert loss.item() > 0
    assert logits.grad is not None
    assert logits.grad[0, 3] != 0
    assert logits.grad[1, 4] != 0


def test_loader_is_seeded_and_rejects_mid_utf8_landmark(tmp_path) -> None:
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "splits": {
                    "train": [
                        {
                            "id": "a",
                            "category": "x",
                            "left": {"text": "猫が走る", "landmark_byte": 3},
                            "right": {"text": "犬が走る", "landmark_byte": 3},
                        },
                        {
                            "id": "b",
                            "category": "x",
                            "left": {"text": "鳥が飛ぶ", "landmark_byte": 3},
                            "right": {"text": "魚が泳ぐ", "landmark_byte": 3},
                        },
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    first, manifest = load_family_consistency_pairs(path, split="train", seed=7)
    second, _ = load_family_consistency_pairs(path, split="train", seed=7)
    assert [pair.pair_id for pair in first] == [pair.pair_id for pair in second]
    assert manifest["records"] == 2

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["splits"]["train"][0]["left"]["landmark_byte"] = 1
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="UTF-8"):
        load_family_consistency_pairs(path, split="train", seed=7)
