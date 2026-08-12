from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_boundary_calibration import (
    ARCHIVE_PREFIX,
    MODEL_CONFIGS,
    PROBES,
    archive_run,
    git_output,
    package_versions,
    ratio_tag,
    run_and_log,
    sha256_file,
    tree_manifest,
)


CHECKPOINT_STEPS = (55, 110, 165, 220)


def run_name(
    main_network: str,
    ratio_weight: float,
    inner_compression_target: float,
    outer_compression_target: float,
    seed: int,
    commit: str,
) -> str:
    return (
        f"r5_match_{main_network}_comp{ratio_tag(inner_compression_target)}-"
        f"{ratio_tag(outer_compression_target)}_rw{ratio_tag(ratio_weight)}_"
        f"utf8hard_s{seed}_step220_{commit[:7]}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one 220-step matched-compression boundary experiment."
    )
    parser.add_argument("--main", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--ratio-weight", type=float, required=True)
    parser.add_argument("--inner-compression-target", type=float, required=True)
    parser.add_argument("--outer-compression-target", type=float, required=True)
    parser.add_argument("--seed", type=int, choices=[42, 43], default=42)
    parser.add_argument("--packed-data-dir", type=Path, required=True)
    parser.add_argument("--packed-validation-data-dir", type=Path, required=True)
    parser.add_argument(
        "--work-root", type=Path, default=Path("/content/hnet_agent_kda_diff_work")
    )
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_PREFIX / "runs")
    parser.add_argument("--dataset-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.packed_data_dir, args.packed_validation_data_dir):
        if not path.is_dir() or not (path / "mix_manifest.json").is_file():
            raise FileNotFoundError(f"Invalid packed dataset directory: {path}")
    archive_root = args.archive_root.resolve()
    if not archive_root.is_relative_to(ARCHIVE_PREFIX):
        raise ValueError(f"Archive root must be under {ARCHIVE_PREFIX}")
    if git_output("branch", "--show-current") != "kimi_attn_diff":
        raise RuntimeError("Training must run on kimi_attn_diff")

    commit = git_output("rev-parse", "HEAD")
    name = run_name(
        args.main,
        args.ratio_weight,
        args.inner_compression_target,
        args.outer_compression_target,
        args.seed,
        commit,
    )
    run_dir = args.work_root / "runs" / name
    archive_dir = archive_root / name
    if run_dir.exists() or archive_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {name}")
    run_dir.mkdir(parents=True)

    if args.dataset_manifest is not None and args.dataset_manifest.exists():
        datasets = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    else:
        datasets = {
            "train": tree_manifest(args.packed_data_dir),
            "validation": tree_manifest(args.packed_validation_data_dir),
        }

    command = [
        sys.executable,
        "train.py",
        "--model-config-path",
        MODEL_CONFIGS[args.main],
        "--output-dir",
        str(run_dir),
        "--packed-data-dir",
        str(args.packed_data_dir),
        "--packed-validation-data-dir",
        str(args.packed_validation_data_dir),
        "--seq-len",
        "2048",
        "--batch-size",
        "16",
        "--grad-accum-steps",
        "32",
        "--max-steps",
        "220",
        "--learning-rate",
        "0.00035",
        "--min-learning-rate",
        "0.00003",
        "--warmup-steps",
        "20",
        "--log-every",
        "5",
        "--save-every",
        "55",
        "--validation-every",
        "55",
        "--validation-max-batches",
        "10",
        "--train-ratio-weight",
        str(args.ratio_weight),
        "--byte-boundary-constraint",
        "utf8-hard",
        "--compression-ratio",
        str(args.inner_compression_target),
        "--compression-ratio",
        str(args.outer_compression_target),
        "--lr-multiplier",
        "2",
        "--lr-multiplier",
        "1.5",
        "--lr-multiplier",
        "1",
        "--seed",
        str(args.seed),
        "--num-workers",
        "0",
    ]
    for prompt in PROBES:
        command.extend(["--chunk-prompt", prompt])

    configuration = {
        "run_name": name,
        "main_network": args.main,
        "ratio_weight": args.ratio_weight,
        "compression_targets": [
            args.inner_compression_target,
            args.outer_compression_target,
        ],
        "seed": args.seed,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "command": command,
    }
    (run_dir / "config.yaml").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    run_and_log(command, run_dir / "training_console.log")

    checkpoints = sorted(run_dir.glob("checkpoint_step_*.pt"))
    expected_names = [f"checkpoint_step_{step:06d}.pt" for step in CHECKPOINT_STEPS]
    if [path.name for path in checkpoints] != expected_names:
        raise RuntimeError(
            f"Expected checkpoints {expected_names}, found {[path.name for path in checkpoints]}"
        )
    manifest = {
        "run_name": name,
        "commit": commit,
        "branch": "kimi_attn_diff",
        "git_dirty": bool(git_output("status", "--porcelain")),
        "model_config": MODEL_CONFIGS[args.main],
        "model_config_sha256": sha256_file(Path(MODEL_CONFIGS[args.main])),
        "datasets": datasets,
        "checkpoints": [
            {
                "path": checkpoint.name,
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
                "archived": True,
            }
            for checkpoint in checkpoints
        ],
        "environment": {
            "python": sys.version,
            "packages": package_versions(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi": subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
                capture_output=True,
                check=False,
            ).stdout.strip(),
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    archive_run(run_dir, archive_dir, include_checkpoint=True)
    print(json.dumps({"run": name, "archive": str(archive_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
