from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_boundary_calibration import (
    ARCHIVE_PREFIX,
    EXPECTED_BRANCH,
    git_output,
    package_versions,
    sha256_file,
)


MODEL_SETTINGS: dict[str, dict[str, object]] = {
    "t26": {
        "config": "configs/phase2/hnet_t26_90m.json",
        "compression": [3.0, 2.5],
        "ratio_weight": 0.08,
        "byte_boundary_constraint": "utf8-hard",
        "lr_multipliers": [2.0, 1.5, 1.0],
    },
    "k1g1": {
        "config": "configs/phase2/hnet_k1g1_90m.json",
        "compression": [3.0, 2.5],
        "ratio_weight": 0.08,
        "byte_boundary_constraint": "utf8-hard",
        "lr_multipliers": [2.0, 1.5, 1.0],
    },
    "k1t1": {
        "config": "configs/phase2/hnet_k1t1_90m.json",
        "compression": [3.0, 3.0],
        "ratio_weight": 0.05,
        "byte_boundary_constraint": "utf8-hard",
        "lr_multipliers": [2.0, 1.5, 1.0],
    },
    "tokenizer": {
        "config": "configs/phase2/tokenizer_t10_128k_90m.json",
        "compression": [1.0],
        "ratio_weight": 0.0,
        "byte_boundary_constraint": "off",
        "lr_multipliers": [1.0],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one raw-byte-budgeted Phase 2 screening experiment."
    )
    parser.add_argument("--model", choices=sorted(MODEL_SETTINGS), required=True)
    parser.add_argument("--seed", type=int, choices=[42, 43], required=True)
    parser.add_argument("--packed-data-dir", type=Path, required=True)
    parser.add_argument("--packed-validation-data-dir", type=Path, required=True)
    parser.add_argument("--max-train-bytes", type=int, default=500_000_000)
    parser.add_argument("--checkpoint-every-bytes", type=int, default=125_000_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--lr-schedule-steps", type=int, default=2000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ARCHIVE_PREFIX / "runs" / "phase2_screening",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    setting = MODEL_SETTINGS[args.model]
    command = [
        sys.executable,
        "train.py",
        "--model-config-path",
        str(setting["config"]),
        "--output-dir",
        str(output_dir),
        "--packed-data-dir",
        str(args.packed_data_dir),
        "--packed-validation-data-dir",
        str(args.packed_validation_data_dir),
        "--seq-len",
        "2048",
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--max-train-bytes",
        str(args.max_train_bytes),
        "--lr-schedule-steps",
        str(args.lr_schedule_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--min-learning-rate",
        str(args.min_learning_rate),
        "--log-every",
        "10",
        "--save-every",
        "0",
        "--save-every-bytes",
        str(args.checkpoint_every_bytes),
        "--validation-every",
        "0",
        "--validation-every-bytes",
        str(args.checkpoint_every_bytes),
        "--validation-max-batches",
        "20",
        "--train-ratio-weight",
        str(setting["ratio_weight"]),
        "--byte-boundary-constraint",
        str(setting["byte_boundary_constraint"]),
        "--seed",
        str(args.seed),
        "--num-workers",
        "0",
    ]
    for ratio in setting["compression"]:
        command.extend(["--compression-ratio", str(ratio)])
    for multiplier in setting["lr_multipliers"]:
        command.extend(["--lr-multiplier", str(multiplier)])
    return command


def main() -> None:
    args = parse_args()
    if git_output("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError(f"Phase 2 must run on {EXPECTED_BRANCH}")
    for path in (args.packed_data_dir, args.packed_validation_data_dir):
        if not path.is_dir() or not (path / "mix_manifest.json").is_file():
            raise FileNotFoundError(f"Invalid packed dataset directory: {path}")
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(ARCHIVE_PREFIX):
        raise ValueError(f"Output root must be under {ARCHIVE_PREFIX}")
    if args.max_train_bytes <= 0 or args.checkpoint_every_bytes <= 0:
        raise ValueError("Byte budgets must be positive")

    commit = git_output("rev-parse", "HEAD")
    data_tag = args.packed_data_dir.name
    run_name = (
        f"{args.model}_95-100m_{data_tag}_ctx2k_s{args.seed}_{commit[:7]}"
    )
    output_dir = output_root / args.model / run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)

    command = build_command(args, output_dir)
    manifest = {
        "run_name": run_name,
        "state": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "branch": EXPECTED_BRANCH,
        "commit": commit,
        "git_dirty": bool(git_output("status", "--porcelain")),
        "model": args.model,
        "model_config": MODEL_SETTINGS[args.model]["config"],
        "model_config_sha256": sha256_file(
            Path(str(MODEL_SETTINGS[args.model]["config"]))
        ),
        "train_mix_manifest_sha256": sha256_file(
            args.packed_data_dir / "mix_manifest.json"
        ),
        "validation_mix_manifest_sha256": sha256_file(
            args.packed_validation_data_dir / "mix_manifest.json"
        ),
        "max_train_bytes": args.max_train_bytes,
        "checkpoint_every_bytes": args.checkpoint_every_bytes,
        "command": command,
        "environment": {
            "python": sys.version,
            "packages": package_versions(),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    cu13_lib = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib"
    env["LD_LIBRARY_PATH"] = f"{cu13_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    with (output_dir / "training_console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    manifest["state"] = "completed" if result.returncode == 0 else "failed"
    manifest["return_code"] = result.returncode
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["checkpoints"] = sorted(
        path.name for path in output_dir.glob("checkpoint_step_*.pt")
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    print(json.dumps({"run": run_name, "output": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
