from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


RUN_PATTERN = re.compile(r"_match_(t26|k1t1|k1g1|m3t1)_.*_s(\d+)_step(\d+)_")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def discover_runs(roots: list[Path]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for root in roots:
        for validation_path in sorted(root.rglob("validation_metrics.csv")):
            run_dir = validation_path.parent
            match = RUN_PATTERN.search(run_dir.name)
            if match is None:
                continue
            validation_rows = read_rows(validation_path)
            training_path = run_dir / "training_metrics.csv"
            if not validation_rows or not training_path.is_file():
                continue
            training_rows = read_rows(training_path)
            if not training_rows:
                continue
            final_validation = validation_rows[-1]
            final_step = int(final_validation["step"])
            main = match.group(1)
            seed = int(match.group(2))
            peak_memory = max(
                float(row["cuda_peak_allocated_mb"]) for row in training_rows
            )
            final_training = training_rows[-1]
            item = {
                "main_network": main,
                "seed": seed,
                "step": final_step,
                "run_dir": str(run_dir),
                "validation_bpb": float(final_validation["validation_bpb"]),
                "validation_ce_loss": float(
                    final_validation["validation_ce_loss"]
                ),
                "compression_l1_l0": float(
                    final_validation["compression_l1_l0"]
                ),
                "compression_l2_l1": float(
                    final_validation["compression_l2_l1"]
                ),
                "compression_l2_l0": float(
                    final_validation["compression_l2_l0"]
                ),
                "input_bytes_per_second": float(
                    final_training["input_bytes_per_second"]
                ),
                "cuda_peak_allocated_mb": peak_memory,
            }
            key = (main, seed)
            previous = selected.get(key)
            if previous is None or final_step >= int(previous["step"]):
                selected[key] = item
    return sorted(selected.values(), key=lambda row: (row["main_network"], row["seed"]))


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for main in sorted({str(row["main_network"]) for row in rows}):
        group = [row for row in rows if row["main_network"] == main]
        bpb = [float(row["validation_bpb"]) for row in group]
        summaries.append(
            {
                "main_network": main,
                "seed_count": len(group),
                "seeds": [int(row["seed"]) for row in group],
                "validation_bpb_mean": statistics.fmean(bpb),
                "validation_bpb_population_sd": statistics.pstdev(bpb),
                "validation_bpb_min": min(bpb),
                "validation_bpb_max": max(bpb),
                "validation_bpb_range": max(bpb) - min(bpb),
                "compression_l2_l0_mean": statistics.fmean(
                    float(row["compression_l2_l0"]) for row in group
                ),
                "input_bytes_per_second_mean": statistics.fmean(
                    float(row["input_bytes_per_second"]) for row in group
                ),
                "cuda_peak_allocated_mb_max": max(
                    float(row["cuda_peak_allocated_mb"]) for row in group
                ),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No Phase 1 rows found")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize matched Phase 1 seeds.")
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = discover_runs(args.run_root)
    summaries = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase1_seed_results.csv", rows)
    write_csv(args.output_dir / "phase1_seed_summary.csv", summaries)
    (args.output_dir / "phase1_seed_summary.json").write_text(
        json.dumps({"runs": rows, "summary": summaries}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
