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
    left_protected_span: tuple[int, int] | None = None
    right_protected_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class FamilyPairBatch:
    input_ids: torch.Tensor
    mask: torch.Tensor
    continuation_mask: torch.Tensor
    landmark_positions: torch.Tensor
    protected_position_mask: torch.Tensor | None = None


def _validate_landmark(text: str, offset: int, *, field: str) -> None:
    encoded = text.encode("utf-8")
    if offset <= 0 or offset >= len(encoded):
        raise ValueError(f"{field} must be inside the encoded text")
    try:
        encoded[:offset].decode("utf-8")
        encoded[offset:].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{field} must be on a UTF-8 character boundary") from error


def _utf8_boundary_offsets(text: str) -> tuple[int, ...]:
    encoded = text.encode("utf-8")
    return tuple(
        offset
        for offset in range(1, len(encoded))
        if not 0x80 <= encoded[offset] <= 0xBF
    )


def _parse_protected_span(
    side: dict[str, Any],
    *,
    text: str,
    landmark: int,
    field: str,
) -> tuple[int, int] | None:
    raw_span = side.get("protected_span")
    if raw_span is None:
        return None
    if not isinstance(raw_span, dict):
        raise ValueError(f"{field} protected_span must be an object")
    start = int(raw_span["start_byte"])
    end = int(raw_span["end_byte"])
    encoded_length = len(text.encode("utf-8"))
    boundaries = set(_utf8_boundary_offsets(text)) | {0, encoded_length}
    if start not in boundaries or end not in boundaries or not 0 <= start < end <= encoded_length:
        raise ValueError(f"{field} protected_span must use valid UTF-8 boundaries")
    if landmark not in {start, end}:
        raise ValueError(f"{field} landmark must border protected_span")
    if not any(start < offset < end for offset in boundaries):
        raise ValueError(f"{field} protected_span must contain an internal boundary")
    return start, end


def load_family_consistency_pairs(
    path: str | Path,
    *,
    split: str,
    seed: int,
    require_protected_span: bool = False,
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
        left_text = str(record["left"]["text"])
        right_text = str(record["right"]["text"])
        left_landmark = int(record["left"]["landmark_byte"])
        right_landmark = int(record["right"]["landmark_byte"])
        left_span = _parse_protected_span(
            record["left"],
            text=left_text,
            landmark=left_landmark,
            field="left",
        )
        right_span = _parse_protected_span(
            record["right"],
            text=right_text,
            landmark=right_landmark,
            field="right",
        )
        if (left_span is None) != (right_span is None):
            raise ValueError("both sides must define protected_span together")
        if require_protected_span and left_span is None:
            raise ValueError("C2 integrity margin requires protected_span on both sides")
        pair = FamilyConsistencyPair(
            pair_id=str(record["id"]),
            category=str(record["category"]),
            left_text=left_text,
            right_text=right_text,
            left_landmark_byte=left_landmark,
            right_landmark_byte=right_landmark,
            left_protected_span=left_span,
            right_protected_span=right_span,
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
        "protected_spans": sum(pair.left_protected_span is not None for pair in pairs),
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
    protected_mask = None
    spans = (pair.left_protected_span, pair.right_protected_span)
    if all(span is not None for span in spans):
        protected_mask = torch.zeros_like(mask)
        for index, span in enumerate(spans):
            assert span is not None
            start, end = span
            for offset in _utf8_boundary_offsets(
                pair.left_text if index == 0 else pair.right_text
            ):
                if start < offset < end:
                    protected_mask[index, offset + 1] = True
    return FamilyPairBatch(
        input_ids,
        mask,
        continuation_mask,
        positions,
        protected_mask,
    )


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


def integrity_margin_loss(
    router_output: RoutingModuleOutput,
    landmark_positions: torch.Tensor,
    protected_position_mask: torch.Tensor | None,
    *,
    margin: float,
) -> torch.Tensor:
    if margin <= 0:
        raise ValueError("C2 margin must be positive")
    probability = router_output.boundary_prob[..., -1].float()
    if probability.ndim != 2 or probability.shape[0] != 2:
        raise ValueError("C2 integrity margin expects exactly one text pair")
    if protected_position_mask is None or protected_position_mask.shape != probability.shape:
        raise ValueError("C2 protected_position_mask must match router probability")
    landmark = probability[
        torch.arange(2, device=probability.device), landmark_positions
    ]
    losses = []
    for index in range(2):
        internal = probability[index][protected_position_mask[index]]
        if internal.numel() == 0:
            raise ValueError("C2 protected span has no internal positions")
        losses.append(F.relu(margin + internal - landmark[index]).mean())
    return torch.stack(losses).mean()
