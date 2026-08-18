from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
import torch.nn.functional as F

from generate import InferenceDType, load_from_pretrained
from hnet.modules.dc import BoundaryOverride, RoutingModuleOutput
from hnet.training.boundary_interventions import (
    evenly_spaced_positions,
    make_boundary_override,
    mask_from_positions,
    morphological_byte_positions,
    prioritized_positions,
    random_positions,
    shifted_positions,
    utf8_safe_positions,
)
from hnet.training.chunk_analysis import percentile
from hnet.training.chunking_utils import render_chunk_content
from hnet.utils.tokenizers import ByteTokenizer


MODES = ("learned", "fixed", "random", "morph", "shifted-left", "shifted-right")


@dataclass(frozen=True)
class ProbeRecord:
    probe_id: str
    category: str
    text: str
    targets: tuple[str, ...] = ()
    requested_prefix_bytes: int | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_utf8_prefix(text: str, requested_bytes: int) -> str:
    raw = text.encode("utf-8")[:requested_bytes]
    while raw:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raw = raw[: exc.start]
    return ""


def load_probe_records(path: Path) -> list[ProbeRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[ProbeRecord] = []
    categories = payload.get("categories", {})
    if not isinstance(categories, dict):
        raise ValueError("probe set 'categories' must be an object")
    global_targets = tuple(str(value) for value in payload.get("targets", []))
    for category, prompts in categories.items():
        if not isinstance(prompts, list):
            raise ValueError(f"category {category!r} must contain a list")
        for index, entry in enumerate(prompts):
            if isinstance(entry, str):
                text = entry
                targets = global_targets
            elif isinstance(entry, dict) and isinstance(entry.get("text"), str):
                text = entry["text"]
                targets = tuple(str(value) for value in entry.get("targets", global_targets))
            else:
                raise ValueError(f"invalid probe entry in category {category!r}")
            records.append(
                ProbeRecord(
                    probe_id=f"{category}-{index:03d}",
                    category=str(category),
                    text=text,
                    targets=targets,
                )
            )

    for doc_index, document in enumerate(payload.get("long_documents", [])):
        if not isinstance(document, dict) or not isinstance(document.get("text"), str):
            raise ValueError("long_documents entries require text")
        source = document["text"] * int(document.get("repeat", 1))
        doc_id = str(document.get("id", f"long-{doc_index:02d}"))
        category = str(document.get("category", "long"))
        for requested in document.get("prefix_bytes", [256, 512, 1024, 2048]):
            prefix = safe_utf8_prefix(source, int(requested))
            if not prefix:
                continue
            records.append(
                ProbeRecord(
                    probe_id=f"{doc_id}-b{int(requested)}",
                    category=category,
                    text=prefix,
                    targets=global_targets,
                    requested_prefix_bytes=int(requested),
                )
            )
    if not records:
        raise ValueError("probe set produced no records")
    return records


def encode_probe(text: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode([text], add_bos=True, add_eos=True)[0][
        "input_ids"
    ].tolist()
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
    labels = torch.tensor(tokens[1:], dtype=torch.long, device=device).unsqueeze(0)
    return input_ids, labels


def selected_positions(mask: torch.Tensor) -> list[int]:
    return [
        int(index)
        for index in torch.nonzero(mask[0], as_tuple=False).flatten().tolist()
    ]


def stage_input_positions(outputs: list[RoutingModuleOutput]) -> list[list[int]]:
    stage0 = selected_positions(outputs[0].boundary_mask)
    stage1_indices = selected_positions(outputs[1].boundary_mask)
    return [stage0, [stage0[index] for index in stage1_indices]]


def proposed_stage_positions(
    mode: str,
    input_ids: torch.Tensor,
    learned: list[RoutingModuleOutput],
    text: str,
    seed: int,
) -> list[list[int]]:
    token_ids = [int(value) for value in input_ids[0].tolist()]
    stage0_learned = selected_positions(learned[0].boundary_mask)
    stage1_learned = selected_positions(learned[1].boundary_mask)
    stage0_candidates = utf8_safe_positions(token_ids)
    stage1_candidates = list(range(len(stage0_learned)))
    counts = [len(stage0_learned), len(stage1_learned)]

    if mode == "learned":
        return [stage0_learned, stage1_learned]
    if mode == "fixed":
        return [
            evenly_spaced_positions(stage0_candidates, counts[0]),
            evenly_spaced_positions(stage1_candidates, counts[1]),
        ]
    if mode == "random":
        return [
            random_positions(stage0_candidates, counts[0], seed),
            random_positions(stage1_candidates, counts[1], seed + 10_000),
        ]
    if mode == "morph":
        morph_positions = morphological_byte_positions(text, add_bos=True)
        stage0 = prioritized_positions(stage0_candidates, morph_positions, counts[0])
        stage1_priority = [
            index for index, position in enumerate(stage0) if position in set(morph_positions)
        ]
        return [
            stage0,
            prioritized_positions(stage1_candidates, stage1_priority, counts[1]),
        ]
    if mode in ("shifted-left", "shifted-right"):
        direction = -1 if mode.endswith("left") else 1
        return [
            shifted_positions(stage0_learned, stage0_candidates, direction),
            shifted_positions(stage1_learned, stage1_candidates, direction),
        ]
    raise ValueError(f"Unsupported boundary mode: {mode}")


def make_overrides(
    learned: list[RoutingModuleOutput],
    positions: list[list[int]],
) -> list[BoundaryOverride]:
    overrides: list[BoundaryOverride] = []
    for output, stage_positions in zip(learned, positions):
        mask = mask_from_positions(
            stage_positions,
            output.boundary_mask.shape[-1],
            output.boundary_mask.device,
        )
        overrides.append(make_boundary_override(output, mask))
    return overrides


def position_to_raw_offset(position: int, add_bos: bool = True) -> int | None:
    if add_bos and position == 0:
        return None
    return position - (1 if add_bos else 0)


def unicode_offset(raw: bytes, byte_offset: int | None) -> int | None:
    if byte_offset is None:
        return None
    return len(raw[:byte_offset].decode("utf-8"))


def chunk_rows(
    record: ProbeRecord,
    mode: str,
    run_seed: int | None,
    outputs: list[RoutingModuleOutput],
    input_ids: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    token_ids = [int(value) for value in input_ids[0].tolist()]
    raw = record.text.encode("utf-8")
    positions_by_stage = stage_input_positions(outputs)
    boundary_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    rendered_lines: list[str] = []

    for stage, positions in enumerate(positions_by_stage):
        chunks: list[list[int]] = []
        char_lengths: list[int] = []
        for boundary_index, start in enumerate(positions):
            end = positions[boundary_index + 1] if boundary_index + 1 < len(positions) else len(token_ids)
            chunk = token_ids[start:end]
            content_tokens = chunk[1:] if start == 0 else chunk
            chunks.append(content_tokens)
            rendered = render_chunk_content(content_tokens)
            char_length = len(bytes(content_tokens).decode("utf-8", errors="replace"))
            char_lengths.append(char_length)
            source_index = start if stage == 0 else selected_positions(outputs[0].boundary_mask).index(start)
            probability = float(outputs[stage].boundary_prob[0, source_index, 1].item())
            raw_offset = position_to_raw_offset(start)
            boundary_rows.append(
                {
                    "probe_id": record.probe_id,
                    "category": record.category,
                    "mode": mode,
                    "random_seed": run_seed,
                    "stage": stage,
                    "boundary_index": boundary_index,
                    "raw_byte_offset": raw_offset,
                    "unicode_char_offset": unicode_offset(raw, raw_offset),
                    "boundary_probability": probability,
                    "chunk_bytes": len(content_tokens),
                    "chunk_chars": char_length,
                    "chunk_text": rendered,
                    "is_sequence_start": start == 0,
                }
            )
        byte_lengths = [len(chunk) for chunk in chunks]
        metric_rows.append(
            {
                "probe_id": record.probe_id,
                "category": record.category,
                "mode": mode,
                "random_seed": run_seed,
                "stage": stage,
                "input_bytes": len(raw),
                "chunk_count": len(chunks),
                "mean_chunk_bytes": mean(byte_lengths),
                "median_chunk_bytes": median(byte_lengths),
                "p90_chunk_bytes": percentile(byte_lengths, 0.90),
                "p95_chunk_bytes": percentile(byte_lengths, 0.95),
                "p99_chunk_bytes": percentile(byte_lengths, 0.99),
                "max_chunk_bytes": max(byte_lengths),
                "mean_chunk_chars": mean(char_lengths),
                "one_char_chunk_rate": sum(value == 1 for value in char_lengths) / len(char_lengths),
                "over_32_byte_chunk_rate": sum(value > 32 for value in byte_lengths) / len(byte_lengths),
                "half_sentence_chunk_rate": sum(value >= max(1, len(raw) / 2) for value in byte_lengths) / len(byte_lengths),
            }
        )
        rendered_lines.append(
            f"stage{stage}: " + " | ".join(render_chunk_content(chunk) for chunk in chunks)
        )
    return boundary_rows, metric_rows, rendered_lines


def boundary_target_rows(
    record: ProbeRecord,
    mode: str,
    run_seed: int | None,
    outputs: list[RoutingModuleOutput],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positions = stage_input_positions(outputs)
    prompt_bytes = record.text.encode("utf-8")
    for target in record.targets:
        target_bytes = target.encode("utf-8")
        cursor = 0
        occurrence = 0
        while True:
            start = prompt_bytes.find(target_bytes, cursor)
            if start < 0:
                break
            end = start + len(target_bytes)
            rows.append(
                {
                    "probe_id": record.probe_id,
                    "mode": mode,
                    "random_seed": run_seed,
                    "target": target,
                    "occurrence": occurrence,
                    "stage0_offsets": [value - 1 - start for value in positions[0] if start + 1 <= value < end + 1],
                    "stage1_offsets": [value - 1 - start for value in positions[1] if start + 1 <= value < end + 1],
                }
            )
            cursor = start + 1
            occurrence += 1
    return rows


def bootstrap_interval(values: list[float], seed: int = 42, samples: int = 2000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    return estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]


def bits_per_raw_byte(token_loss: torch.Tensor, raw_bytes: int) -> float:
    if raw_bytes <= 0:
        raise ValueError("raw_bytes must be positive")
    return float(token_loss.sum().item()) / (raw_bytes * math.log(2.0))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate learned and counterfactual H-Net chunk boundaries."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--probe-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", action="append", choices=MODES, dest="modes")
    parser.add_argument("--random-seed", action="append", type=int, dest="random_seeds")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--utf8-hard", action="store_true")
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Skip per-probe chunk and next-byte files while retaining aggregate artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = args.modes or list(MODES)
    random_seeds = args.random_seeds or [0, 1, 2, 3, 4]
    records = load_probe_records(args.probe_set)
    if args.max_records is not None:
        records = records[: args.max_records]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = args.output_dir / "validation_chunks"
    prediction_dir = args.output_dir / "probe_predictions"
    if not args.compact:
        chunk_dir.mkdir(exist_ok=True)
        prediction_dir.mkdir(exist_ok=True)

    model = load_from_pretrained(
        str(args.model_path), str(args.config_path), requested_dtype=args.dtype
    )
    device = next(model.parameters()).device
    boundary_rows: list[dict[str, Any]] = []
    chunk_metric_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    profile_accumulator: dict[tuple[str, int, int, str, int | None], list[float]] = defaultdict(list)
    target_rows: list[dict[str, Any]] = []

    for record_index, record in enumerate(records):
        input_ids, labels = encode_probe(record.text, device)
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        continuation = (input_ids >= 0x80) & (input_ids <= 0xBF)
        with torch.inference_mode():
            learned_result = model(
                input_ids=input_ids,
                mask=mask,
                continuation_mask=continuation if args.utf8_hard else None,
                continuation_hard=args.utf8_hard,
            )
        learned_outputs = list(learned_result.bpred_output)
        if len(learned_outputs) != 2:
            raise ValueError("Boundary intervention evaluation currently requires two stages")
        learned_positions = proposed_stage_positions(
            "learned", input_ids, learned_outputs, record.text, seed=0
        )
        learned_overrides = make_overrides(learned_outputs, learned_positions)
        with torch.inference_mode():
            identity_result = model(
                input_ids=input_ids,
                mask=mask,
                continuation_mask=continuation if args.utf8_hard else None,
                continuation_hard=args.utf8_hard,
                boundary_overrides=learned_overrides,
            )
        if not torch.allclose(
            learned_result.logits, identity_result.logits, atol=1e-5, rtol=1e-5
        ):
            max_diff = float((learned_result.logits - identity_result.logits).abs().max())
            raise RuntimeError(f"Learned-boundary identity check failed: max_diff={max_diff}")

        reference_bpb: float | None = None
        learned_boundary_positions = stage_input_positions(learned_outputs)
        for mode in modes:
            seeds: list[int | None] = random_seeds if mode == "random" else [None]
            for run_seed in seeds:
                positions = proposed_stage_positions(
                    mode,
                    input_ids,
                    learned_outputs,
                    record.text,
                    seed=run_seed or 0,
                )
                if mode == "learned":
                    result = learned_result
                else:
                    overrides = make_overrides(learned_outputs, positions)
                    with torch.inference_mode():
                        result = model(
                            input_ids=input_ids,
                            mask=mask,
                            continuation_mask=continuation if args.utf8_hard else None,
                            continuation_hard=args.utf8_hard,
                            boundary_overrides=overrides,
                        )
                token_loss = F.cross_entropy(
                    result.logits[0].float(), labels[0], reduction="none"
                )
                ce = float(token_loss.mean().item())
                raw_bytes = len(record.text.encode("utf-8"))
                bpb = bits_per_raw_byte(token_loss, raw_bytes)
                if mode == "learned":
                    reference_bpb = bpb
                if reference_bpb is None:
                    reference_bpb = bits_per_raw_byte(
                        F.cross_entropy(
                            learned_result.logits[0].float(),
                            labels[0],
                            reduction="none",
                        ),
                        raw_bytes,
                    )
                evaluation_rows.append(
                    {
                        "probe_id": record.probe_id,
                        "category": record.category,
                        "mode": mode,
                        "random_seed": run_seed,
                        "input_bytes": raw_bytes,
                        "ce_loss": ce,
                        "bpb": bpb,
                        "delta_bpb": bpb - reference_bpb,
                        "stage0_chunk_count": int(result.bpred_output[0].boundary_mask.sum().item()),
                        "stage1_chunk_count": int(result.bpred_output[1].boundary_mask.sum().item()),
                    }
                )
                new_boundary_rows, new_metric_rows, rendered = chunk_rows(
                    record, mode, run_seed, list(result.bpred_output), input_ids
                )
                boundary_rows.extend(new_boundary_rows)
                chunk_metric_rows.extend(new_metric_rows)
                target_rows.extend(
                    boundary_target_rows(record, mode, run_seed, list(result.bpred_output))
                )
                chunk_name = f"{record.probe_id}_{mode}"
                if run_seed is not None:
                    chunk_name += f"_s{run_seed}"
                if not args.compact:
                    (chunk_dir / f"{chunk_name}.txt").write_text(
                        record.text + "\n" + "\n".join(rendered) + "\n",
                        encoding="utf-8",
                    )
                    top_ids = torch.topk(
                        result.logits[0, -1].float(), k=5
                    ).indices.tolist()
                    (prediction_dir / f"{chunk_name}.json").write_text(
                        json.dumps({"next_byte_top5": top_ids}, indent=2) + "\n",
                        encoding="utf-8",
                    )
                for stage, stage_positions in enumerate(learned_boundary_positions):
                    for boundary_position in stage_positions:
                        for relative in range(-4, 9):
                            loss_index = boundary_position + relative
                            if 0 <= loss_index < token_loss.numel():
                                profile_accumulator[
                                    (record.category, stage, relative, mode, run_seed)
                                ].append(float(token_loss[loss_index].item()))
        print(f"[{record_index + 1}/{len(records)}] {record.probe_id}", flush=True)

    with (args.output_dir / "boundary_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in boundary_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "target_boundary_patterns.jsonl").open("w", encoding="utf-8") as handle:
        for row in target_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(args.output_dir / "chunk_metrics.csv", chunk_metric_rows)
    write_csv(args.output_dir / "counterfactual_boundary_eval.csv", evaluation_rows)
    profile_rows = [
        {
            "category": key[0],
            "stage": key[1],
            "relative_byte_offset": key[2],
            "mode": key[3],
            "random_seed": key[4],
            "mean_ce_loss": mean(values),
            "observations": len(values),
        }
        for key, values in sorted(profile_accumulator.items(), key=lambda item: str(item[0]))
    ]
    write_csv(args.output_dir / "boundary_loss_profile.csv", profile_rows)

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in evaluation_rows:
        grouped[(str(row["category"]), str(row["mode"]))].append(float(row["delta_bpb"]))
    summary_rows: list[dict[str, Any]] = []
    for (category, mode), values in sorted(grouped.items()):
        ci_low, ci_high = bootstrap_interval(values)
        summary_rows.append(
            {
                "category": category,
                "mode": mode,
                "observations": len(values),
                "mean_delta_bpb": mean(values),
                "std_delta_bpb": stdev(values) if len(values) > 1 else 0.0,
                "bootstrap_95_low": ci_low,
                "bootstrap_95_high": ci_high,
            }
        )
    write_csv(args.output_dir / "counterfactual_boundary_summary.csv", summary_rows)

    manifest = {
        "commit": git_commit(),
        "model_path": str(args.model_path),
        "model_sha256": sha256_file(args.model_path),
        "config_path": str(args.config_path),
        "config_sha256": sha256_file(args.config_path),
        "probe_set": str(args.probe_set),
        "probe_set_sha256": sha256_file(args.probe_set),
        "record_count": len(records),
        "modes": modes,
        "random_seeds": random_seeds,
        "utf8_hard": args.utf8_hard,
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "learned_boundary_identity_check": "passed",
        "bpb_normalization": "sum NLL / raw UTF-8 bytes / ln(2)",
        "compact": args.compact,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
