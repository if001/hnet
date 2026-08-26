from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hnet.training.linguistic_boundary_families import (
    FamilyLandmark,
    landmark_byte_offset,
)


PRIMARY_LIMITS = {
    "category_fracture_record_occupancy": ("max_delta", 0.05),
    "family_coverage": ("min_delta", -0.05),
    "landmark_coverage": ("min_delta", -0.05),
    "family_integrity": ("min_delta", -0.05),
}


@dataclass(frozen=True)
class RunInput:
    label: str
    raw_dir: Path
    control_label: str | None


def parse_run(value: str) -> RunInput:
    parts = value.split("=", 2)
    if len(parts) not in {2, 3} or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError(
            "--run must be LABEL=RAW_DIR or LABEL=RAW_DIR=CONTROL_LABEL"
        )
    return RunInput(parts[0], Path(parts[1]), parts[2] if len(parts) == 3 else None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize full112 factor trajectories and control constraints."
    )
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("stage0", "stage1"), default="stage1")
    parser.add_argument("--condition", default="native")
    return parser.parse_args()


def checkpoint_step(payload: dict[str, Any], path: Path) -> int:
    text = str(payload.get("checkpoint_label", path.stem))
    match = re.search(r"step[_-]?(\d+)", text, re.IGNORECASE)
    if match is None:
        raise ValueError(f"cannot identify step for {path}")
    return int(match.group(1))


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_records(
    records: Iterable[dict[str, Any]], *, condition: str, stage: str
) -> dict[str, float | int | None]:
    materialized = list(records)
    scores = [record["conditions"][condition][stage]["score"] for record in materialized]
    selected = sum(len(score["selected_evaluable_offsets"]) for score in scores)
    explained = sum(len(score["explained_offsets"]) for score in scores)
    acceptable = sum(len(score["acceptable_offsets"]) for score in scores)
    fractured = sum(bool(score.get("lexeme_fracture_offsets")) for score in scores)
    return {
        "records": len(scores),
        "precision": safe_ratio(explained, selected),
        "coverage": safe_ratio(explained, acceptable),
        "fracture_record_occupancy": safe_ratio(fractured, len(scores)),
        "integrity": safe_ratio(len(scores) - fractured, len(scores)),
    }


def aggregate_landmarks(
    records: Iterable[dict[str, Any]], *, condition: str, stage: str
) -> float | None:
    selected_count = 0
    opportunities = 0
    for record in records:
        family = record.get("family")
        if family is None:
            continue
        selected = set(
            record["conditions"][condition][stage]["score"][
                "selected_evaluable_offsets"
            ]
        )
        surface = record["focus"]["surface"]
        for item in family["landmarks"]:
            landmark = FamilyLandmark(item["name"], item["segmentation"])
            selected_count += landmark_byte_offset(surface, landmark) in selected
            opportunities += 1
    return safe_ratio(selected_count, opportunities)


def summarize_payload(
    label: str,
    payload: dict[str, Any],
    path: Path,
    *,
    condition: str,
    stage: str,
) -> dict[str, Any]:
    records = payload["records"]
    category_records = [record for record in records if record.get("family") is None]
    family_records = [record for record in records if record.get("family") is not None]
    bunsetsu_records = [
        record for record in category_records if record["category"] == "bunsetsu"
    ]
    if len(category_records) != 88 or len(family_records) != 24:
        raise ValueError(
            f"{path}: expected category88+family24, got "
            f"{len(category_records)}+{len(family_records)}"
        )
    category = aggregate_records(category_records, condition=condition, stage=stage)
    family = aggregate_records(family_records, condition=condition, stage=stage)
    bunsetsu = aggregate_records(bunsetsu_records, condition=condition, stage=stage)
    return {
        "label": label,
        "step": checkpoint_step(payload, path),
        "cumulative_input_bytes": payload.get("cumulative_input_bytes"),
        "category_precision": category["precision"],
        "category_coverage": category["coverage"],
        "category_fracture_record_occupancy": category[
            "fracture_record_occupancy"
        ],
        "family_precision": family["precision"],
        "family_coverage": family["coverage"],
        "landmark_coverage": aggregate_landmarks(
            family_records, condition=condition, stage=stage
        ),
        "family_integrity": family["integrity"],
        "bunsetsu_precision": bunsetsu["precision"],
        "bunsetsu_coverage": bunsetsu["coverage"],
        "source_path": str(path),
    }


def load_rows(
    run: RunInput, *, condition: str, stage: str
) -> list[dict[str, Any]]:
    paths = sorted(run.raw_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no JSON files in {run.raw_dir}")
    rows = [
        summarize_payload(
            run.label,
            json.loads(path.read_text(encoding="utf-8")),
            path,
            condition=condition,
            stage=stage,
        )
        for path in paths
    ]
    if len({row["step"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate steps for {run.label}")
    return sorted(rows, key=lambda row: row["step"])


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def window_name(step: int, max_step: int) -> str:
    if max_step <= 100:
        if step <= 30:
            return "early"
        if step <= 60:
            return "middle"
        return "late"
    if step <= 50:
        return "early"
    if step <= 120:
        return "middle"
    if step <= 170:
        return "late"
    return "terminal"


def window_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        key
        for key in rows[0]
        if key not in {"label", "step", "cumulative_input_bytes", "source_path"}
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    max_step = max(row["step"] for row in rows)
    for row in rows:
        grouped[window_name(row["step"], max_step)].append(row)
    output = []
    for name, group in grouped.items():
        for metric in metrics:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            med = median(values)
            output.append(
                {
                    "label": rows[0]["label"],
                    "window": name,
                    "start_step": min(row["step"] for row in group),
                    "end_step": max(row["step"] for row in group),
                    "metric": metric,
                    "mean": mean(values),
                    "median": med,
                    "q20": quantile(values, 0.2),
                    "q80": quantile(values, 0.8),
                    "mad": median([abs(value - med) for value in values]),
                    "range": max(values) - min(values),
                }
            )
    return output


def constraint_rows(
    variant: list[dict[str, Any]], control: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    control_by_step = {row["step"]: row for row in control}
    output = []
    for row in variant:
        baseline = control_by_step.get(row["step"])
        if baseline is None:
            continue
        for metric, (direction, limit) in PRIMARY_LIMITS.items():
            delta = float(row[metric]) - float(baseline[metric])
            violated = delta > limit if direction == "max_delta" else delta < limit
            output.append(
                {
                    "label": row["label"],
                    "control_label": baseline["label"],
                    "step": row["step"],
                    "metric": metric,
                    "variant": row[metric],
                    "control": baseline[metric],
                    "delta": delta,
                    "primary_limit": limit,
                    "violated": violated,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    labels = [run.label for run in args.run]
    if len(set(labels)) != len(labels):
        raise ValueError("run labels must be unique")
    all_rows = {
        run.label: load_rows(run, condition=args.condition, stage=args.stage)
        for run in args.run
    }
    constraints = []
    for run in args.run:
        if run.control_label is None:
            continue
        if run.control_label not in all_rows:
            raise ValueError(f"missing control label: {run.control_label}")
        constraints.extend(
            constraint_rows(all_rows[run.label], all_rows[run.control_label])
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [row for rows in all_rows.values() for row in rows]
    windows = [item for rows in all_rows.values() for item in window_rows(rows)]
    write_csv(args.output_dir / "per_step_metrics.csv", metrics)
    write_csv(args.output_dir / "window_summary.csv", windows)
    write_csv(args.output_dir / "control_constraints.csv", constraints)
    summary = {
        "stage": args.stage,
        "condition": args.condition,
        "labels": labels,
        "steps": {label: [row["step"] for row in rows] for label, rows in all_rows.items()},
        "constraint_violations": {
            label: sum(row["violated"] for row in constraints if row["label"] == label)
            for label in labels
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
