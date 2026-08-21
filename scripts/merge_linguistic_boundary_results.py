from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


MATCH_FIELDS = (
    "model_name",
    "model_path",
    "config_path",
    "seed",
    "byte_boundary_constraint",
    "byte_boundary_constraint_bias",
    "budget_profiles",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge matched category and family boundary results."
    )
    parser.add_argument("--category-root", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--category-records", type=int, default=88)
    parser.add_argument("--family-records", type=int, default=24)
    return parser.parse_args()


def checkpoint_step(label: object) -> int:
    match = re.search(r"(?:^|[^a-z])step[_-]?(\d+)", str(label), re.IGNORECASE)
    if match is None:
        raise ValueError(f"checkpoint label has no step: {label!r}")
    return int(match.group(1))


def result_key(payload: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(payload["model_name"]),
        int(payload["seed"]),
        checkpoint_step(payload["checkpoint_label"]),
    )


def load_result_map(root: Path) -> dict[tuple[str, int, int], tuple[Path, dict[str, Any]]]:
    results: dict[tuple[str, int, int], tuple[Path, dict[str, Any]]] = {}
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "records" not in payload:
            continue
        key = result_key(payload)
        if key in results:
            raise ValueError(f"duplicate result key {key}: {path}")
        results[key] = (path, payload)
    if not results:
        raise FileNotFoundError(f"no boundary result JSON under {root}")
    return results


def merge_payloads(
    category: dict[str, Any],
    family: dict[str, Any],
    *,
    expected_category_records: int,
    expected_family_records: int,
) -> dict[str, Any]:
    category_key = result_key(category)
    family_key = result_key(family)
    if category_key != family_key:
        raise ValueError(f"result key mismatch: {category_key} != {family_key}")
    mismatches = {
        field: (category.get(field), family.get(field))
        for field in MATCH_FIELDS
        if category.get(field) != family.get(field)
    }
    if mismatches:
        raise ValueError(f"result metadata mismatch for {category_key}: {mismatches}")
    category_records = category["records"]
    family_records = family["records"]
    if len(category_records) != expected_category_records:
        raise ValueError(
            f"category record count is {len(category_records)}, "
            f"expected {expected_category_records}"
        )
    if len(family_records) != expected_family_records:
        raise ValueError(
            f"family record count is {len(family_records)}, "
            f"expected {expected_family_records}"
        )
    category_ids = {record["id"] for record in category_records}
    family_ids = {record["id"] for record in family_records}
    overlap = category_ids & family_ids
    if overlap:
        raise ValueError(f"record IDs overlap: {sorted(overlap)}")
    model_name, seed, step = category_key
    return {
        "version": max(int(category["version"]), int(family["version"])),
        "model_name": model_name,
        "model_path": category["model_path"],
        "config_path": category["config_path"],
        "seed": seed,
        "checkpoint_label": f"step{step}-combined112-v1",
        "probe_path": "category88+family24",
        "probe_paths": {
            "category": category.get("probe_path"),
            "family": family.get("probe_path"),
        },
        "byte_boundary_constraint": category["byte_boundary_constraint"],
        "byte_boundary_constraint_bias": category[
            "byte_boundary_constraint_bias"
        ],
        "budget_profiles": category["budget_profiles"],
        "record_sets": {
            "category": len(category_records),
            "family": len(family_records),
            "total": len(category_records) + len(family_records),
        },
        "records": [*category_records, *family_records],
    }


def main() -> None:
    args = parse_args()
    category_results = load_result_map(args.category_root)
    family_results = load_result_map(args.family_root)
    if set(category_results) != set(family_results):
        raise ValueError(
            "category/family result key sets differ: "
            f"category_only={sorted(set(category_results) - set(family_results))}, "
            f"family_only={sorted(set(family_results) - set(category_results))}"
        )
    output_paths = []
    for key in sorted(category_results):
        category_path, category = category_results[key]
        family_path, family = family_results[key]
        merged = merge_payloads(
            category,
            family,
            expected_category_records=args.category_records,
            expected_family_records=args.family_records,
        )
        model_name, seed, step = key
        model_dir = args.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        output_path = model_dir / (
            f"{model_name}_s{seed}_step{step}_combined112_v1.json"
        )
        merged["source_paths"] = {
            "category": str(category_path),
            "family": str(family_path),
        }
        output_path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_paths.append(output_path)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "files": len(output_paths),
                "records_per_file": args.category_records + args.family_records,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
