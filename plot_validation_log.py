import argparse
import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit("matplotlib is required. Install dependencies with `pip install -e .`.") from exc


NUMERIC_FIELDS = [
    "step",
    "validation_ce_loss",
    "validation_bpb",
    "avg_bytes_per_chunk",
    "target_ratio_gap",
    "actual_selected_fraction",
    "mean_boundary_probability",
]


def load_validation_metrics(csv_path: Path) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {key: [] for key in NUMERIC_FIELDS}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key in NUMERIC_FIELDS:
                metrics[key].append(float(row[key]))
    return metrics


def plot_validation_metrics(csv_path: Path, output_path: Path) -> None:
    metrics = load_validation_metrics(csv_path)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(metrics["step"], metrics["validation_ce_loss"], label="val_ce")
    axes[0].plot(metrics["step"], metrics["validation_bpb"], label="val_bpb")
    axes[0].set_ylabel("quality")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(metrics["step"], metrics["avg_bytes_per_chunk"], label="avg_bytes/chunk")
    axes[1].plot(metrics["step"], metrics["actual_selected_fraction"], label="selected_fraction")
    axes[1].plot(metrics["step"], metrics["mean_boundary_probability"], label="mean_boundary_prob")
    axes[1].set_ylabel("chunking")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(metrics["step"], metrics["target_ratio_gap"], color="tab:red", label="target_ratio_gap")
    axes[2].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("gap")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot H-Net validation metrics from CSV")
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to validation_metrics.csv",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts/validation_metrics.png",
        help="Path to the output PNG image",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_path = Path(args.output_path)
    plot_validation_metrics(csv_path, output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
