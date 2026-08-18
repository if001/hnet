import csv
import json

from scripts.summarize_phase2_results import (
    aggregate_curves,
    aggregate_runs,
    collect_pretraining,
    extract_sft_losses,
)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_collect_and_aggregate_phase2_metrics(tmp_path):
    run = tmp_path / "t26" / "t26_example_s42_commit"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"seed": 42}), encoding="utf-8")
    write_csv(
        run / "training_metrics.csv",
        [
            {"step": 1, "elapsed_seconds": 2, "cumulative_input_bytes": 100,
             "cuda_peak_allocated_mb": 10},
            {"step": 2, "elapsed_seconds": 4, "cumulative_input_bytes": 200,
             "cuda_peak_allocated_mb": 12},
        ],
    )
    write_csv(
        run / "validation_metrics.csv",
        [
            {"step": 1, "cumulative_input_bytes": 100, "validation_bpb": 2.0,
             "compression_l2_l0": 8, "stage0_mid_utf8_boundary_fraction": 0},
            {"step": 2, "cumulative_input_bytes": 200, "validation_bpb": 1.8,
             "compression_l2_l0": 9, "stage0_mid_utf8_boundary_fraction": 0},
        ],
    )
    for step in (1, 2):
        (run / f"checkpoint_step_{step:06d}.pt").touch()

    runs, curves = collect_pretraining(tmp_path)
    assert runs[0]["validation_bpb"] == 1.8
    assert runs[0]["effective_raw_bytes_per_second"] == 50
    assert runs[0]["checkpoint_count"] == 2
    assert aggregate_runs(runs)[0]["seed_count"] == 1
    assert aggregate_curves(curves)[1]["validation_bpb_mean"] == 1.8


def test_extract_sft_losses_ignores_non_metric_lines():
    text = "sample {'role': 'user'}\n{'loss': '4.2', 'epoch': '0.5'}\n"
    assert extract_sft_losses(text) == [{"loss": "4.2", "epoch": "0.5"}]
