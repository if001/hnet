import csv

import pytest

from scripts.summarize_phase1_selection import aggregate, discover_runs


def _write_run(root, name: str, bpb: float, step: int = 220) -> None:
    run = root / name
    run.mkdir(parents=True)
    with (run / "validation_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "validation_bpb",
                "validation_ce_loss",
                "compression_l1_l0",
                "compression_l2_l1",
                "compression_l2_l0",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "step": step,
                "validation_bpb": bpb,
                "validation_ce_loss": 1.0,
                "compression_l1_l0": 3.0,
                "compression_l2_l1": 2.5,
                "compression_l2_l0": 7.5,
            }
        )
    with (run / "training_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["input_bytes_per_second", "cuda_peak_allocated_mb"],
        )
        writer.writeheader()
        writer.writerow(
            {"input_bytes_per_second": 1000.0, "cuda_peak_allocated_mb": 2000.0}
        )


def test_phase1_summary_keeps_best_step_per_seed(tmp_path) -> None:
    _write_run(tmp_path, "r5_match_t26_x_s42_step55_abc", 2.0, step=55)
    _write_run(tmp_path, "r5_match_t26_x_s42_step220_def", 1.6)
    _write_run(tmp_path, "r5_match_t26_x_s43_step220_def", 1.8)

    rows = discover_runs([tmp_path])
    summary = aggregate(rows)

    assert [row["seed"] for row in rows] == [42, 43]
    assert rows[0]["step"] == 220
    assert summary[0]["validation_bpb_mean"] == pytest.approx(1.7)
    assert summary[0]["validation_bpb_range"] == pytest.approx(0.2)
