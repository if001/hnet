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

from scripts.run_boundary_calibration import ARCHIVE_PREFIX, EXPECTED_BRANCH, git_output
from scripts.run_phase2_screening import MODEL_SETTINGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run common Phase 2 agent-proxy SFT.")
    parser.add_argument("--model", choices=sorted(MODEL_SETTINGS), required=True)
    parser.add_argument("--pretrained-model-path", type=Path, required=True)
    parser.add_argument("--pretraining-seed", type=int, choices=[42, 43], required=True)
    parser.add_argument("--sft-seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--mix-config-path",
        type=Path,
        default=Path("hnet/sft/configs/phase2_agent_proxy.json"),
    )
    parser.add_argument(
        "--model-tokenizer-path",
        type=Path,
        default=ARCHIVE_PREFIX / "tokenizers" / "128k" / "tokenizer.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ARCHIVE_PREFIX / "runs" / "phase2_sft_proxy",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    setting = MODEL_SETTINGS[args.model]
    command = [
        sys.executable,
        "-m",
        "hnet.sft.train",
        "--model-config-path",
        str(setting["config"]),
        "--pretrained-model-path",
        str(args.pretrained_model_path),
        "--output-dir",
        str(output_dir),
        "--mix-config-path",
        str(args.mix_config_path),
        "--seq-len",
        str(args.seq_len),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--learning-rate",
        "0.0001",
        "--warmup-steps",
        "20",
        "--logging-steps",
        "5",
        "--save-steps",
        "100",
        "--train-ratio-weight",
        str(setting["ratio_weight"]),
        "--seed",
        str(args.sft_seed),
    ]
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    if args.model == "tokenizer":
        command.extend(["--model-tokenizer-path", str(args.model_tokenizer_path)])
    for ratio in setting["compression"]:
        command.extend(["--compression-ratio", str(ratio)])
    for multiplier in setting["lr_multipliers"]:
        command.extend(["--lr-multiplier", str(multiplier)])
    return command


def main() -> None:
    args = parse_args()
    if git_output("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError(f"Phase 2 SFT must run on {EXPECTED_BRANCH}")
    if not args.pretrained_model_path.is_file():
        raise FileNotFoundError(args.pretrained_model_path)
    if not args.mix_config_path.is_file():
        raise FileNotFoundError(args.mix_config_path)
    if args.model == "tokenizer" and not args.model_tokenizer_path.is_file():
        raise FileNotFoundError(args.model_tokenizer_path)
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(ARCHIVE_PREFIX):
        raise ValueError(f"Output root must be under {ARCHIVE_PREFIX}")

    commit = git_output("rev-parse", "HEAD")
    run_name = (
        f"{args.model}_pre_s{args.pretraining_seed}_sft_s{args.sft_seed}_"
        f"ctx{args.seq_len}_{commit[:7]}"
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
        "model": args.model,
        "pretraining_seed": args.pretraining_seed,
        "sft_seed": args.sft_seed,
        "command": command,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    cu13_lib = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib"
    env["LD_LIBRARY_PATH"] = f"{cu13_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    with (output_dir / "training_console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    manifest["state"] = "completed" if result.returncode == 0 else "failed"
    manifest["return_code"] = result.returncode
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


if __name__ == "__main__":
    main()
