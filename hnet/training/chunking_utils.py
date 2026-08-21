from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from hnet.utils.tokenizers import ByteTokenizer


@contextmanager
def deterministic_model_evaluation(model: Any):
    """Run evaluation without changing model mode or process RNG streams."""
    was_training = bool(model.training)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model.eval()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        model.train(was_training)


def decode_bytes(token_ids: list[int]) -> str:
    return bytes(token_ids).decode("utf-8", errors="replace")


def render_chunk_content(token_ids: list[int]) -> str:
    """Render chunk as mixed UTF-8 text and raw bytes.

    Decodable UTF-8 spans are shown as plain text, undecodable bytes are shown as <0xHH>.
    """
    raw = bytes(token_ids)
    cursor = 0
    parts: list[str] = []

    while cursor < len(raw):
        current = raw[cursor:]
        try:
            parts.append(current.decode("utf-8"))
            break
        except UnicodeDecodeError as exc:
            if exc.start > 0:
                valid_prefix = current[: exc.start].decode("utf-8")
                parts.append(valid_prefix)

            bad = current[exc.start : exc.end]
            if bad:
                parts.extend(f"<0x{value:02X}>" for value in bad)
                cursor += exc.end
            else:
                parts.append(f"<0x{current[0]:02X}>")
                cursor += 1

    rendered = "".join(parts)
    return rendered if rendered else "<empty>"


def format_chunk_compact(token_ids: list[int]) -> str:
    mixed = render_chunk_content(token_ids)
    return f"[{mixed}]"


def format_stage_compact(chunks: list[list[int]]) -> str:
    return ", ".join(format_chunk_compact(chunk) for chunk in chunks)


def make_byte_chunks(token_ids: list[int], boundary_mask: list[bool]) -> list[list[int]]:
    boundary_positions = [idx for idx, is_boundary in enumerate(boundary_mask) if is_boundary]
    if not boundary_positions:
        return []

    chunks: list[list[int]] = []
    for i, start in enumerate(boundary_positions):
        end = boundary_positions[i + 1] if i + 1 < len(boundary_positions) else len(token_ids)
        chunks.append(token_ids[start:end])
    return chunks


@torch.no_grad()
def inspect_prompt_chunks(
    model: Any,
    prompt: str,
    add_bos: bool = True,
    utf8_hard: bool = False,
) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    prompt_utf8_bytes = list(prompt.encode("utf-8"))
    encoded = tokenizer.encode([prompt], add_bos=add_bos)[0]["input_ids"]
    token_ids = [int(x) for x in encoded.tolist()]

    device = next(model.parameters()).device
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)

    continuation_mask = (input_ids >= 0x80) & (input_ids <= 0xBF)
    output = model(
        input_ids=input_ids,
        mask=mask,
        continuation_mask=continuation_mask if utf8_hard else None,
        continuation_hard=utf8_hard,
    )
    if len(output.bpred_output) < 2:
        raise ValueError(
            "This utility expects a 2-stage model, but boundary outputs are fewer than 2."
        )

    stage0_mask_tensor = output.bpred_output[0].boundary_mask[0].detach().cpu()
    stage1_mask_tensor = output.bpred_output[1].boundary_mask[0].detach().cpu()
    stage0_probability = (
        output.bpred_output[0].boundary_prob[0, :, 1].detach().float().cpu().tolist()
    )
    stage1_probability = (
        output.bpred_output[1].boundary_prob[0, :, 1].detach().float().cpu().tolist()
    )

    stage0_mask = [bool(v) for v in stage0_mask_tensor.tolist()]
    stage1_mask = [bool(v) for v in stage1_mask_tensor.tolist()]

    stage0_boundaries = [idx for idx, is_boundary in enumerate(stage0_mask) if is_boundary]
    stage0_chunks = make_byte_chunks(token_ids, stage0_mask)

    stage1_source_positions = stage0_boundaries
    if len(stage1_mask) > len(stage1_source_positions):
        stage1_mask = stage1_mask[: len(stage1_source_positions)]

    stage1_boundary_indices_in_stage0 = [
        idx for idx, is_boundary in enumerate(stage1_mask) if is_boundary
    ]
    stage1_boundary_positions_in_input = [
        stage1_source_positions[idx] for idx in stage1_boundary_indices_in_stage0
    ]

    stage1_chunks: list[list[int]] = []
    if stage1_boundary_positions_in_input:
        for i, start in enumerate(stage1_boundary_positions_in_input):
            end = (
                stage1_boundary_positions_in_input[i + 1]
                if i + 1 < len(stage1_boundary_positions_in_input)
                else len(token_ids)
            )
            stage1_chunks.append(token_ids[start:end])

    return {
        "prompt": prompt,
        "prompt_utf8_bytes": prompt_utf8_bytes,
        "token_ids": token_ids,
        "add_bos": add_bos,
        "stage0_mask": stage0_mask,
        "stage0_boundary_probability": stage0_probability,
        "stage1_mask": stage1_mask,
        "stage1_boundary_probability": stage1_probability,
        "stage0_boundaries": stage0_boundaries,
        "stage0_chunks": stage0_chunks,
        "stage1_boundary_indices_in_stage0": stage1_boundary_indices_in_stage0,
        "stage1_boundary_positions_in_input": stage1_boundary_positions_in_input,
        "stage1_chunks": stage1_chunks,
    }
