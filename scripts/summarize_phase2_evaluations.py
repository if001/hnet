from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_from_name(name: str) -> int:
    for part in name.split("_"):
        if part.startswith("s") and part[1:].isdigit():
            return int(part[1:])
        if part.startswith("pre_s") and part[5:].isdigit():
            return int(part[5:])
    marker = "_pre_s"
    if marker in name:
        return int(name.split(marker, 1)[1].split("_", 1)[0])
    raise ValueError(f"Cannot infer seed from {name}")


def collect_general(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            path = run_dir / "text_bpb_summary.csv"
            if not path.exists():
                continue
            seed = seed_from_name(run_dir.name)
            for row in read_csv(path):
                rows.append(
                    {
                        "model": model_dir.name,
                        "seed": seed,
                        "run": run_dir.name,
                        "category": row["category"],
                        "records": int(row["records"]),
                        "raw_bytes": int(row["raw_bytes"]),
                        "micro_bpb": float(row["micro_bpb"]),
                        "record_macro_bpb": float(row["record_macro_bpb"]),
                    }
                )
    return rows


def collect_boundary(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            path = run_dir / "counterfactual_boundary_eval.csv"
            if not path.exists():
                continue
            source = read_csv(path)
            grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
            for row in source:
                grouped[(row["category"], row["mode"])].append(float(row["delta_bpb"]))
                grouped[("__overall__", row["mode"])].append(float(row["delta_bpb"]))
            seed = seed_from_name(run_dir.name)
            for (category, mode), values in sorted(grouped.items()):
                rows.append(
                    {
                        "model": model_dir.name,
                        "seed": seed,
                        "run": run_dir.name,
                        "category": category,
                        "mode": mode,
                        "observations": len(values),
                        "mean_delta_bpb": statistics.fmean(values),
                    }
                )
    return rows


def collect_transfer(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for direction in sorted(path for path in root.iterdir() if path.is_dir()):
        for seed_dir in sorted(path for path in direction.iterdir() if path.is_dir()):
            path = seed_dir / "boundary_transfer_summary.csv"
            if not path.exists():
                continue
            for row in read_csv(path):
                rows.append(
                    {
                        "direction": direction.name,
                        "seed": int(seed_dir.name.removeprefix("s")),
                        "category": row["category"],
                        "records": int(row["records"]),
                        "mean_delta_bpb": float(row["mean_delta_bpb"]),
                        "bootstrap_95_low": float(row["bootstrap_95_low"]),
                        "bootstrap_95_high": float(row["bootstrap_95_high"]),
                    }
                )
    return rows


def collect_agent(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            path = run_dir / "agent_proxy_summary.csv"
            if not path.exists():
                continue
            seed = seed_from_name(run_dir.name)
            for row in read_csv(path):
                rows.append(
                    {
                        "model": model_dir.name,
                        "pretraining_seed": seed,
                        "run": run_dir.name,
                        "category": row["category"],
                        "tasks": int(row["tasks"]),
                        "json_valid_rate": float(row["json_valid_rate"]),
                        "tool_accuracy": float(row["tool_accuracy"]),
                        "argument_exact_rate": float(row["argument_exact_rate"]),
                        "full_exact_rate": float(row["full_exact_rate"]),
                    }
                )
    return rows


def aggregate_model_category(
    rows: list[dict[str, Any]], value_keys: list[str], seed_key: str = "seed"
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["category"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, category), values in sorted(grouped.items()):
        result: dict[str, Any] = {
            "model": model,
            "category": category,
            "seed_count": len({int(row[seed_key]) for row in values}),
        }
        for key in value_keys:
            numeric = [float(row[key]) for row in values]
            result[f"{key}_mean"] = statistics.fmean(numeric)
            result[f"{key}_population_sd"] = statistics.pstdev(numeric)
        output.append(result)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate completed Phase 2 evaluations.")
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    general = collect_general(args.target_root / "evals/general_phase2")
    boundary = collect_boundary(args.target_root / "evals/phase2_boundary")
    transfer = collect_transfer(args.target_root / "evals/phase2_transfer")
    agent = collect_agent(args.target_root / "evals/agent_phase2")
    general_summary = aggregate_model_category(
        general, ["micro_bpb", "record_macro_bpb"]
    )
    agent_summary = aggregate_model_category(
        agent,
        ["json_valid_rate", "tool_accuracy", "argument_exact_rate", "full_exact_rate"],
        seed_key="pretraining_seed",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "general_records": general,
        "general_summary": general_summary,
        "boundary_summary": boundary,
        "transfer_summary": transfer,
        "agent_records": agent,
        "agent_summary": agent_summary,
    }
    for name, rows in outputs.items():
        write_csv(args.output_dir / f"{name}.csv", rows)
    (args.output_dir / "phase2_evaluation_summary.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
