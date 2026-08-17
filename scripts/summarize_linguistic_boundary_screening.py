from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate linguistic boundary screening JSON files."
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_runs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    runs = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError(f"unsupported result version in {path}")
        payload["_source_path"] = str(path)
        runs.append(payload)
    return runs


def optional_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def aggregate_scores(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    metadata: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for run in runs:
        run_id = f"{run['model_name']}|{run.get('seed')}|{run.get('checkpoint_label')}"
        for record in run["records"]:
            for condition, condition_result in record["conditions"].items():
                for stage in ("stage0", "stage1"):
                    key = (run_id, condition, stage, record["category"])
                    groups[key].append(condition_result[stage]["score"])
                    metadata[key] = {
                        "model_name": run["model_name"],
                        "seed": run.get("seed"),
                        "checkpoint_label": run.get("checkpoint_label"),
                        "constraint": run["byte_boundary_constraint"],
                        "source_path": run["_source_path"],
                    }

    rows = []
    for key, scores in sorted(groups.items()):
        run_id, condition, stage, category = key
        selected = sum(len(score["selected_evaluable_offsets"]) for score in scores)
        explained = sum(len(score["explained_offsets"]) for score in scores)
        unexplained = sum(len(score["unexplained_offsets"]) for score in scores)
        acceptable = sum(len(score["acceptable_offsets"]) for score in scores)
        constraint_dependent = sum(
            len(score["constraint_dependent_offsets"]) for score in scores
        )
        pathological = sum(
            bool(score["has_pathological_fragmentation"]) for score in scores
        )
        row = {
            **metadata[key],
            "run_id": run_id,
            "condition": condition,
            "stage": stage,
            "category": category,
            "records": len(scores),
            "selected_evaluable_boundaries": selected,
            "explained_boundaries": explained,
            "unexplained_boundaries": unexplained,
            "acceptable_boundary_candidates": acceptable,
            "constraint_dependent_boundaries": constraint_dependent,
            "explainable_boundary_precision": explained / selected
            if selected
            else None,
            "category_coverage": explained / acceptable if acceptable else None,
            "unexplained_boundary_rate": unexplained / selected
            if selected
            else None,
            "pathological_fragmentation_record_rate": pathological / len(scores),
        }
        rows.append(row)
    return rows


def pair_scores(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        run_id = f"{run['model_name']}|{run.get('seed')}|{run.get('checkpoint_label')}"
        condition_names = run["records"][0]["conditions"].keys()
        for condition in condition_names:
            for stage in ("stage0", "stage1"):
                grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
                for record in run["records"]:
                    pair = record.get("pair")
                    if not pair:
                        continue
                    score = record["conditions"][condition][stage]["score"]
                    grouped[(pair["group"], pair["kind"])].append(score)
                for (group, kind), scores in sorted(grouped.items()):
                    signatures = {
                        tuple(score["selected_evaluable_offsets"]) for score in scores
                    }
                    no_unexplained = all(
                        not score["unexplained_offsets"] for score in scores
                    )
                    any_explained = any(score["explained_offsets"] for score in scores)
                    if kind == "control":
                        passed = len(signatures) == 1
                    elif kind == "dynamic":
                        passed = len(signatures) > 1 and no_unexplained and any_explained
                    else:
                        raise ValueError(f"unknown pair kind: {kind}")
                    rows.append(
                        {
                            "model_name": run["model_name"],
                            "seed": run.get("seed"),
                            "checkpoint_label": run.get("checkpoint_label"),
                            "run_id": run_id,
                            "condition": condition,
                            "stage": stage,
                            "pair_group": group,
                            "pair_kind": kind,
                            "records": len(scores),
                            "distinct_signatures": len(signatures),
                            "no_unexplained": no_unexplained,
                            "any_explained": any_explained,
                            "passed": passed,
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def masked_label(index: int) -> str:
    return f"candidate-{index + 1:02d}"


def write_gallery(path: Path, runs: list[dict[str, Any]]) -> None:
    ordered = sorted(
        runs,
        key=lambda run: (
            run["model_name"],
            run.get("seed") if run.get("seed") is not None else -1,
            run.get("checkpoint_label") or "",
        ),
    )
    lines = [
        "# Linguistic boundary blind gallery",
        "",
        "Candidate mapping is intentionally listed only at the end of the file.",
        "Primary inspection condition: `central`.",
        "",
    ]
    record_ids = [record["id"] for record in ordered[0]["records"]]
    for record_id in record_ids:
        exemplar = next(
            record for record in ordered[0]["records"] if record["id"] == record_id
        )
        lines.extend(
            [
                f"## {record_id}",
                "",
                f"Category: `{exemplar['category']}`",
                "",
                exemplar["text"],
                "",
            ]
        )
        for index, run in enumerate(ordered):
            record = next(item for item in run["records"] if item["id"] == record_id)
            central = record["conditions"]["central"]
            lines.extend(
                [
                    f"### {masked_label(index)}",
                    "",
                    f"- stage0: `{central['stage0']['chunks']}`",
                    f"- stage1: `{central['stage1']['chunks']}`",
                    "",
                ]
            )
    lines.extend(["## Candidate mapping", ""])
    for index, run in enumerate(ordered):
        lines.append(
            f"- {masked_label(index)}: {run['model_name']} / seed={run.get('seed')} "
            f"/ checkpoint={run.get('checkpoint_label')}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = load_runs(args.input)
    score_rows = aggregate_scores(runs)
    pair_rows = pair_scores(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "linguistic_boundary_scores.csv", score_rows)
    write_csv(args.output_dir / "linguistic_boundary_pairs.csv", pair_rows)
    write_gallery(args.output_dir / "linguistic_boundary_blind_gallery.md", runs)
    summary = {
        "version": 1,
        "input_count": len(runs),
        "inputs": [run["_source_path"] for run in runs],
        "score_rows": score_rows,
        "pair_rows": pair_rows,
    }
    (args.output_dir / "linguistic_boundary_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
