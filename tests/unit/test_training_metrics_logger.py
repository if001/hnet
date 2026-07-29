import csv

from hnet.training.trainer import TrainingMetricsLogger


def test_training_metrics_logger_writes_performance_fields(tmp_path) -> None:
    output_path = tmp_path / "training_metrics.csv"
    logger = TrainingMetricsLogger(output_path)

    logger.log(
        step=1,
        learning_rate=3.5e-4,
        ce_loss=1.5,
        ratio_loss=0.2,
        byte_boundary_loss=0.01,
        total_loss=1.5045,
        elapsed_seconds=12.0,
        step_seconds=10.0,
        input_bytes=1_048_576,
        input_bytes_per_second=104_857.6,
        cuda_peak_allocated_mb=24_000.0,
    )

    with output_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["elapsed_seconds"] == "12.0"
    assert rows[0]["step_seconds"] == "10.0"
    assert rows[0]["input_bytes"] == "1048576"
    assert rows[0]["input_bytes_per_second"] == "104857.6"
    assert rows[0]["cuda_peak_allocated_mb"] == "24000.0"


def test_training_metrics_logger_migrates_old_header_on_resume(tmp_path) -> None:
    output_path = tmp_path / "training_metrics.csv"
    output_path.write_text(
        "step,learning_rate,ce_loss,ratio_loss,byte_boundary_loss,total_loss\n"
        "4,0.00035,1.5,0.2,0.01,1.5045\n",
        encoding="utf-8",
    )

    TrainingMetricsLogger(output_path, append=True)

    with output_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["step"] == "4"
    assert rows[0]["elapsed_seconds"] == ""
    assert rows[0]["input_bytes_per_second"] == ""
