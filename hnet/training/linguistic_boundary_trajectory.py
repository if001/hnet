from __future__ import annotations

import re
from collections import defaultdict
from statistics import mean
from typing import Any


def checkpoint_order(label: object) -> int:
    if isinstance(label, int):
        return label
    step_match = re.search(r"(?:^|[^a-z])step[_-]?(\d+)", str(label), re.IGNORECASE)
    if step_match:
        return int(step_match.group(1))
    matches = re.findall(r"\d+", str(label))
    if not matches:
        raise ValueError(f"checkpoint label has no numeric order: {label!r}")
    return int(matches[-1])


def _micro_metrics(scores: list[dict[str, Any]]) -> dict[str, float | None]:
    selected = sum(len(score["selected_evaluable_offsets"]) for score in scores)
    explained = sum(len(score["explained_offsets"]) for score in scores)
    acceptable = sum(len(score["acceptable_offsets"]) for score in scores)
    return {
        "precision": explained / selected if selected else None,
        "coverage": explained / acceptable if acceptable else None,
        "unexplained_record_occupancy": sum(
            bool(score["unexplained_offsets"]) for score in scores
        )
        / len(scores),
        "fracture_record_occupancy": sum(
            bool(score.get("lexeme_fracture_offsets")) for score in scores
        )
        / len(scores),
    }


def summarize_trajectory_scores(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate sparse or dense checkpoint sequences without assuming convergence."""
    groups: dict[
        tuple[str, object, str, str, str],
        dict[int, dict[str, dict[str, Any]]],
    ] = defaultdict(lambda: defaultdict(dict))

    for run in runs:
        step = checkpoint_order(run.get("checkpoint_label"))
        for record in run["records"]:
            dimensions = ("__all__", record["category"])
            for condition, result in record["conditions"].items():
                for stage in ("stage0", "stage1"):
                    for dimension in dimensions:
                        key = (
                            run["model_name"],
                            run.get("seed"),
                            condition,
                            stage,
                            dimension,
                        )
                        if record["id"] in groups[key][step]:
                            raise ValueError(
                                "duplicate trajectory observation: "
                                f"{key}, {step}, {record['id']}"
                            )
                        groups[key][step][record["id"]] = result[stage]["score"]

    rows: list[dict[str, Any]] = []
    for key, checkpoints in sorted(groups.items(), key=lambda item: str(item[0])):
        if len(checkpoints) < 2:
            continue
        model_name, seed, condition, stage, dimension = key
        steps = sorted(checkpoints)
        all_scores = [
            score
            for step in steps
            for score in checkpoints[step].values()
        ]
        late_steps = steps[-min(2, len(steps)) :]
        late_scores = [
            score
            for step in late_steps
            for score in checkpoints[step].values()
        ]
        transitions = 0
        comparisons = 0
        for previous, current in zip(steps, steps[1:]):
            common = set(checkpoints[previous]) & set(checkpoints[current])
            for record_id in common:
                previous_signature = tuple(
                    checkpoints[previous][record_id]["selected_evaluable_offsets"]
                )
                current_signature = tuple(
                    checkpoints[current][record_id]["selected_evaluable_offsets"]
                )
                transitions += previous_signature != current_signature
                comparisons += 1
        full = _micro_metrics(all_scores)
        late = _micro_metrics(late_scores)
        per_step_precision = []
        per_step_coverage = []
        for step in steps:
            metrics = _micro_metrics(list(checkpoints[step].values()))
            if metrics["precision"] is not None:
                per_step_precision.append(float(metrics["precision"]))
            if metrics["coverage"] is not None:
                per_step_coverage.append(float(metrics["coverage"]))
        rows.append(
            {
                "model_name": model_name,
                "seed": seed,
                "condition": condition,
                "stage": stage,
                "category": dimension,
                "checkpoint_count": len(steps),
                "first_checkpoint": steps[0],
                "last_checkpoint": steps[-1],
                "late_checkpoint_count": len(late_steps),
                "time_averaged_precision": mean(per_step_precision)
                if per_step_precision
                else None,
                "time_averaged_coverage": mean(per_step_coverage)
                if per_step_coverage
                else None,
                "unexplained_record_occupancy": full["unexplained_record_occupancy"],
                "fracture_record_occupancy": full["fracture_record_occupancy"],
                "late_precision": late["precision"],
                "late_coverage": late["coverage"],
                "late_unexplained_record_occupancy": late[
                    "unexplained_record_occupancy"
                ],
                "late_fracture_record_occupancy": late[
                    "fracture_record_occupancy"
                ],
                "segmentation_transition_rate": transitions / comparisons
                if comparisons
                else None,
                "transition_comparisons": comparisons,
            }
        )
    return rows
