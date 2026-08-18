from __future__ import annotations

import argparse
import ast
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


def infer_seed(run_dir: Path) -> int:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("seed", "pretraining_seed"):
            if key in manifest:
                return int(manifest[key])
    for part in run_dir.name.split("_"):
        if part.startswith("s") and part[1:].isdigit():
            return int(part[1:])
    raise ValueError(f"Cannot infer seed from {run_dir}")


def collect_pretraining(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        model = model_dir.name
        for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            training_path = run_dir / "training_metrics.csv"
            validation_path = run_dir / "validation_metrics.csv"
            if not training_path.exists() or not validation_path.exists():
                continue
            training = read_csv(training_path)
            validation = read_csv(validation_path)
            if not training or not validation:
                continue
            seed = infer_seed(run_dir)
            final_train = training[-1]
            final_validation = validation[-1]
            elapsed = float(final_train["elapsed_seconds"])
            cumulative = int(final_train["cumulative_input_bytes"])
            run_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "run": run_dir.name,
                    "final_step": int(final_train["step"]),
                    "cumulative_input_bytes": cumulative,
                    "validation_bpb": float(final_validation["validation_bpb"]),
                    "compression_l2_l0": float(final_validation["compression_l2_l0"]),
                    "effective_raw_bytes_per_second": cumulative / elapsed,
                    "peak_cuda_allocated_mb": max(
                        float(row["cuda_peak_allocated_mb"]) for row in training
                    ),
                    "checkpoint_count": len(list(run_dir.glob("checkpoint_step_*.pt"))),
                }
            )
            for checkpoint_index, row in enumerate(validation, start=1):
                curve_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "run": run_dir.name,
                        "checkpoint_index": checkpoint_index,
                        "step": int(row["step"]),
                        "cumulative_input_bytes": int(row["cumulative_input_bytes"]),
                        "validation_bpb": float(row["validation_bpb"]),
                        "compression_l2_l0": float(row["compression_l2_l0"]),
                        "stage0_mid_utf8_boundary_fraction": float(
                            row["stage0_mid_utf8_boundary_fraction"]
                        ),
                    }
                )
    return run_rows, curve_rows


def aggregate_runs(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["model"])].append(row)
    summaries: list[dict[str, Any]] = []
    for model, rows in sorted(grouped.items()):
        bpb = [float(row["validation_bpb"]) for row in rows]
        throughput = [float(row["effective_raw_bytes_per_second"]) for row in rows]
        memory = [float(row["peak_cuda_allocated_mb"]) for row in rows]
        compression = [float(row["compression_l2_l0"]) for row in rows]
        summaries.append(
            {
                "model": model,
                "seed_count": len(rows),
                "seeds": ",".join(str(row["seed"]) for row in sorted(rows, key=lambda x: x["seed"])),
                "validation_bpb_mean": statistics.fmean(bpb),
                "validation_bpb_population_sd": statistics.pstdev(bpb),
                "validation_bpb_range": max(bpb) - min(bpb),
                "compression_l2_l0_mean": statistics.fmean(compression),
                "effective_raw_bytes_per_second_mean": statistics.fmean(throughput),
                "peak_cuda_allocated_mb_max": max(memory),
            }
        )
    return summaries


def aggregate_curves(curve_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        grouped[(str(row["model"]), int(row["checkpoint_index"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, checkpoint_index), rows in sorted(grouped.items()):
        output.append(
            {
                "model": model,
                "checkpoint_index": checkpoint_index,
                "seed_count": len(rows),
                "cumulative_input_bytes_mean": statistics.fmean(
                    float(row["cumulative_input_bytes"]) for row in rows
                ),
                "validation_bpb_mean": statistics.fmean(
                    float(row["validation_bpb"]) for row in rows
                ),
                "validation_bpb_population_sd": statistics.pstdev(
                    float(row["validation_bpb"]) for row in rows
                ),
                "compression_l2_l0_mean": statistics.fmean(
                    float(row["compression_l2_l0"]) for row in rows
                ),
            }
        )
    return output


def extract_sft_losses(log_text: str) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        start = line.find("{'loss':")
        if start < 0:
            continue
        try:
            payload = ast.literal_eval(line[start:])
        except (SyntaxError, ValueError):
            continue
        if isinstance(payload, dict) and "loss" in payload:
            metrics.append(payload)
    return metrics


def collect_sft(sft_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not sft_root.exists():
        return rows
    for model_dir in sorted(path for path in sft_root.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            if "_failed_" in run_dir.name:
                continue
            manifest_path = run_dir / "manifest.json"
            log_path = run_dir / "training_console.log"
            if not manifest_path.exists() or not log_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            losses = extract_sft_losses(log_path.read_text(encoding="utf-8", errors="replace"))
            rows.append(
                {
                    "model": model_dir.name,
                    "pretraining_seed": int(manifest["pretraining_seed"]),
                    "sft_seed": int(manifest["sft_seed"]),
                    "run": run_dir.name,
                    "state": manifest["state"],
                    "logged_loss_count": len(losses),
                    "final_logged_loss": float(losses[-1]["loss"]) if losses else float("nan"),
                    "final_model_exists": (run_dir / "sft_final_model.pt").is_file(),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Phase 2 pretraining and SFT runs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sft-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_rows, curve_rows = collect_pretraining(args.run_root)
    run_summary = aggregate_runs(run_rows)
    curve_summary = aggregate_curves(curve_rows)
    sft_rows = collect_sft(args.sft_root)
    write_csv(args.output_dir / "phase2_run_metrics.csv", run_rows)
    write_csv(args.output_dir / "phase2_curve_metrics.csv", curve_rows)
    write_csv(args.output_dir / "phase2_seed_summary.csv", run_summary)
    write_csv(args.output_dir / "phase2_curve_summary.csv", curve_summary)
    write_csv(args.output_dir / "phase2_sft_summary.csv", sft_rows)
    payload = {
        "run_metrics": run_rows,
        "seed_summary": run_summary,
        "curve_summary": curve_summary,
        "sft_summary": sft_rows,
    }
    (args.output_dir / "phase2_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
