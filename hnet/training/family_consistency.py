from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hnet.modules.dc import RoutingModuleOutput


BOS_TOKEN_ID = 0xFE


@dataclass(frozen=True)
class FamilyConsistencyPair:
    pair_id: str
    category: str
    left_text: str
    right_text: str
    left_landmark_byte: int
    right_landmark_byte: int


@dataclass(frozen=True)
class FamilyPairBatch:
    input_ids: torch.Tensor
    mask: torch.Tensor
    continuation_mask: torch.Tensor
    landmark_positions: torch.Tensor


def _validate_landmark(text: str, offset: int, *, field: str) -> None:
    encoded = text.encode("utf-8")
    if offset <= 0 or offset >= len(encoded):
        raise ValueError(f"{field} must be inside the encoded text")
    try:
        encoded[:offset].decode("utf-8")
        encoded[offset:].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{field} must be on a UTF-8 character boundary") from error


def load_family_consistency_pairs(
    path: str | Path,
    *,
    split: str,
    seed: int,
) -> tuple[list[FamilyConsistencyPair], dict[str, Any]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("family consistency dataset version must be 1")
    raw_records = payload.get("splits", {}).get(split)
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"family consistency split is empty: {split}")
    pairs = []
    ids = set()
    for record in raw_records:
        pair = FamilyConsistencyPair(
            pair_id=str(record["id"]),
            category=str(record["category"]),
            left_text=str(record["left"]["text"]),
            right_text=str(record["right"]["text"]),
            left_landmark_byte=int(record["left"]["landmark_byte"]),
            right_landmark_byte=int(record["right"]["landmark_byte"]),
        )
        if not pair.pair_id or pair.pair_id in ids:
            raise ValueError("family consistency pair ids must be non-empty and unique")
        ids.add(pair.pair_id)
        _validate_landmark(pair.left_text, pair.left_landmark_byte, field="left landmark")
        _validate_landmark(pair.right_text, pair.right_landmark_byte, field="right landmark")
        pairs.append(pair)
    random.Random(seed).shuffle(pairs)
    metadata = {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "split": split,
        "seed": seed,
        "records": len(pairs),
        "categories": sorted({pair.category for pair in pairs}),
        "ordered_pair_ids": [pair.pair_id for pair in pairs],
    }
    return pairs, metadata


def build_family_pair_batch(
    pair: FamilyConsistencyPair,
    *,
    device: torch.device,
) -> FamilyPairBatch:
    encoded = [pair.left_text.encode("utf-8"), pair.right_text.encode("utf-8")]
    lengths = [len(value) + 1 for value in encoded]
    max_length = max(lengths)
    input_ids = torch.zeros((2, max_length), dtype=torch.long, device=device)
    mask = torch.zeros((2, max_length), dtype=torch.bool, device=device)
    for index, value in enumerate(encoded):
        tokens = torch.tensor([BOS_TOKEN_ID, *value], dtype=torch.long, device=device)
        input_ids[index, : tokens.numel()] = tokens
        mask[index, : tokens.numel()] = True
    continuation_mask = (
        (input_ids >= 0x80) & (input_ids <= 0xBF) & mask
    )
    positions = torch.tensor(
        [pair.left_landmark_byte + 1, pair.right_landmark_byte + 1],
        dtype=torch.long,
        device=device,
    )
    return FamilyPairBatch(input_ids, mask, continuation_mask, positions)


def landmark_consistency_loss(
    router_output: RoutingModuleOutput,
    landmark_positions: torch.Tensor,
) -> torch.Tensor:
    probability = router_output.boundary_prob[..., -1].float()
    if probability.ndim != 2 or probability.shape[0] != 2:
        raise ValueError("C1 landmark consistency expects exactly one text pair")
    if landmark_positions.shape != (2,):
        raise ValueError("landmark_positions must have shape (2,)")
    if torch.any(landmark_positions < 0) or torch.any(
        landmark_positions >= probability.shape[1]
    ):
        raise ValueError("landmark position is outside router output")
    selected = probability[
        torch.arange(2, device=probability.device), landmark_positions
    ]
    return F.mse_loss(selected[0], selected[1])
