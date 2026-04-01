import argparse
import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit("matplotlib is required. Install dependencies with `pip install -e .`.") from exc


def load_metrics(csv_path: Path) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {
        "step": [],
        "learning_rate": [],
        "ce_loss": [],
        "ratio_loss": [],
        "total_loss": [],
    }
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key in metrics:
                metrics[key].append(float(row[key]))
    return metrics


def plot_metrics(csv_path: Path, output_path: Path) -> None:
    metrics = load_metrics(csv_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(metrics["step"], metrics["total_loss"], label="total_loss")
    axes[0].plot(metrics["step"], metrics["ce_loss"], label="ce_loss")
    axes[0].plot(metrics["step"], metrics["ratio_loss"], label="ratio_loss")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(metrics["step"], metrics["learning_rate"], color="tab:orange", label="learning_rate")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("lr")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot H-Net training metrics from CSV")
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to training_metrics.csv",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts/training_metrics.png",
        help="Path to the output PNG image",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_path = Path(args.output_path)
    plot_metrics(csv_path, output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
