import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from hnet.training.chunk_analysis import (
    mid_utf8_boundary_rate,
    summarize_lengths,
    target_boundary_offsets,
)
from hnet.training.chunking_utils import inspect_prompt_chunks
from inspect_chunking import load_from_pretrained

def load_eval_set(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts", [])
    targets = payload.get("targets", [])
    if not isinstance(prompts, list) or not all(isinstance(x, str) for x in prompts):
        raise ValueError("eval set 'prompts' must be a list of strings")
    if not isinstance(targets, list) or not all(isinstance(x, str) for x in targets):
        raise ValueError("eval set 'targets' must be a list of strings")
    return prompts, targets


def analyze(
    model: Any,
    prompts: list[str],
    targets: list[str],
    utf8_hard: bool = False,
) -> dict[str, Any]:
    stage_chunks: list[list[list[int]]] = [[], []]
    midpoint_rates: list[list[float]] = [[], []]
    target_patterns: dict[str, list[dict[str, Any]]] = {target: [] for target in targets}
    prompt_results: list[dict[str, Any]] = []

    for prompt in prompts:
        info = inspect_prompt_chunks(
            model, prompt, add_bos=True, utf8_hard=utf8_hard
        )
        stage_positions = [
            info["stage0_boundaries"],
            info["stage1_boundary_positions_in_input"],
        ]
        chunks = [info["stage0_chunks"], info["stage1_chunks"]]
        prompt_result: dict[str, Any] = {"prompt": prompt}
        for stage_index in range(2):
            stage_chunks[stage_index].extend(chunks[stage_index])
            rate = mid_utf8_boundary_rate(info["token_ids"], stage_positions[stage_index])
            midpoint_rates[stage_index].append(rate)
            prompt_result[f"stage{stage_index}_chunk_lengths"] = [
                len(chunk) for chunk in chunks[stage_index]
            ]
            prompt_result[f"stage{stage_index}_mid_utf8_boundary_rate"] = rate

        for target in targets:
            if target not in prompt:
                continue
            target_patterns[target].append(
                {
                    "prompt": prompt,
                    "stage0_offsets": target_boundary_offsets(
                        prompt, target, stage_positions[0]
                    ),
                    "stage1_offsets": target_boundary_offsets(
                        prompt, target, stage_positions[1]
                    ),
                }
            )
        prompt_results.append(prompt_result)

    summary: dict[str, Any] = {}
    for stage_index in range(2):
        summary[f"stage{stage_index}"] = {
            "chunk_lengths": summarize_lengths(stage_chunks[stage_index]),
            "mean_mid_utf8_boundary_rate": mean(midpoint_rates[stage_index])
            if midpoint_rates[stage_index]
            else float("nan"),
        }
    summary["target_boundary_patterns"] = target_patterns
    summary["prompts"] = prompt_results
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze H-Net chunk distributions and context-dependent boundaries."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--utf8-hard",
        action="store_true",
        help="Disallow stage-0 boundaries on UTF-8 continuation bytes.",
    )
    args = parser.parse_args()

    prompts, targets = load_eval_set(args.eval_set)
    model = load_from_pretrained(args.model_path, args.config_path)
    result = analyze(model, prompts, targets, utf8_hard=args.utf8_hard)
    result["model_path"] = args.model_path
    result["config_path"] = args.config_path
    result["eval_set"] = str(args.eval_set)
    result["utf8_hard"] = args.utf8_hard
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
