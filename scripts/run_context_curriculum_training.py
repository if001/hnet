from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_boundary_calibration import (
    ARCHIVE_PREFIX,
    EXPECTED_BRANCH,
    MODEL_CONFIGS,
    archive_run,
    git_output,
    run_and_log,
    sha256_file,
)
from scripts.run_dense_linguistic_training import (
    OUTER_COMPRESSION_TARGETS,
    chunk_prompt_arg,
    load_probe_prompts,
)


BASE_SEQ_LEN = 32768
BYTES_PER_UPDATE = 1024 * 1024
TOTAL_STEPS = 600
TOTAL_TRAIN_BYTES = TOTAL_STEPS * BYTES_PER_UPDATE


@dataclass(frozen=True)
class CurriculumLeg:
    name: str
    schedule: str
    phase: str
    seq_len: int
    max_steps: int
    expected_resume_step: int | None
    transition_steps: tuple[int, ...] = ()

    @property
    def batch_size(self) -> int:
        return BASE_SEQ_LEN // self.seq_len


LEGS = {
    "shared-a": CurriculumLeg("shared-a", "shared", "A", 2048, 200, None),
    "l0-tail": CurriculumLeg("l0-tail", "L0", "tail", 2048, 600, 200),
    "l1-b": CurriculumLeg("l1-b", "L1", "B", 8192, 400, 200, (201, 205)),
    "l1-c": CurriculumLeg("l1-c", "L1", "C", 32768, 600, 400, (401, 405)),
    "l2-c": CurriculumLeg("l2-c", "L2", "C", 32768, 600, 200, (201, 205)),
    "l3-full": CurriculumLeg("l3-full", "L3", "full", 32768, 600, None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one aligned 2K/8K/32K context-curriculum leg."
    )
    parser.add_argument("--main", choices=("t26", "k1g1", "k3g1"), required=True)
    parser.add_argument("--leg", choices=sorted(LEGS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-init-seed", type=int, default=42)
    parser.add_argument("--data-order-seed", type=int, default=42)
    parser.add_argument("--train-runtime-seed", type=int, default=42)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--initial-model-checkpoint", type=Path)
    parser.add_argument("--save-initial-model-to", type=Path)
    parser.add_argument("--packed-data-dir", type=Path, required=True)
    parser.add_argument("--packed-validation-data-dir", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument(
        "--work-root", type=Path, default=Path("/content/hnet_context_work")
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=ARCHIVE_PREFIX / "runs/context_curriculum_p6_v1",
    )
    parser.add_argument("--run-prefix", default="r21_p6")
    return parser.parse_args()


def latest_checkpoint(run_dir: Path) -> tuple[Path, int]:
    checkpoints = sorted(run_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint in resume run: {run_dir}")
    checkpoint = checkpoints[-1]
    return checkpoint, int(checkpoint.stem.split("_")[-1])


def seed_factors(args: argparse.Namespace) -> dict[str, int]:
    return {
        "model_init_seed": int(args.model_init_seed),
        "data_order_seed": int(args.data_order_seed),
        "train_runtime_seed": int(args.train_runtime_seed),
    }


def validate_resume(args: argparse.Namespace, leg: CurriculumLeg) -> Path | None:
    if leg.expected_resume_step is None:
        if args.resume_run_dir is not None:
            raise ValueError(f"{leg.name} must start from step 0")
        return None
    if args.resume_run_dir is None:
        raise ValueError(f"{leg.name} requires --resume-run-dir")
    checkpoint, step = latest_checkpoint(args.resume_run_dir)
    if step != leg.expected_resume_step:
        raise ValueError(
            f"{leg.name} requires step {leg.expected_resume_step}, got {step}"
        )
    metadata_path = args.resume_run_dir / "context_run_config.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("main") != args.main:
        raise ValueError("resume main-network mismatch")
    if metadata.get("seed_factors") != seed_factors(args):
        raise ValueError("resume seed-factor mismatch")
    if int(metadata.get("base_seq_len", -1)) != BASE_SEQ_LEN:
        raise ValueError("resume canonical context mismatch")
    if int(metadata.get("total_train_bytes", -1)) != TOTAL_TRAIN_BYTES:
        raise ValueError("resume raw-byte horizon mismatch")
    return checkpoint


def build_train_command(
    args: argparse.Namespace,
    leg: CurriculumLeg,
    run_dir: Path,
    resume_checkpoint: Path | None,
    prompts: list[str],
) -> list[str]:
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
        "--packed-curriculum-base-seq-len",
        str(BASE_SEQ_LEN),
        "--seq-len",
        str(leg.seq_len),
        "--batch-size",
        str(leg.batch_size),
        "--grad-accum-steps",
        "32",
        "--max-steps",
        str(leg.max_steps),
        "--max-train-bytes",
        str(TOTAL_TRAIN_BYTES),
        "--lr-schedule-steps",
        str(TOTAL_STEPS),
        "--learning-rate",
        "0.00035",
        "--min-learning-rate",
        "0.00003",
        "--warmup-steps",
        "20",
        "--log-every",
        "5",
        "--save-every",
        "100",
        "--validation-every",
        "10",
        "--validation-max-batches",
        "0",
        "--train-ratio-weight",
        "0.08",
        "--byte-boundary-constraint",
        "utf8-hard",
        "--compression-ratio",
        "3.0",
        "--compression-ratio",
        str(OUTER_COMPRESSION_TARGETS[args.main]),
        "--lr-multiplier",
        "2",
        "--lr-multiplier",
        "1.5",
        "--lr-multiplier",
        "1",
        "--seed",
        str(args.seed),
        "--model-init-seed",
        str(args.model_init_seed),
        "--data-order-seed",
        str(args.data_order_seed),
        "--train-runtime-seed",
        str(args.train_runtime_seed),
        "--num-workers",
        "0",
    ]
    for step in leg.transition_steps:
        command.extend(["--validation-step", str(step)])
    if resume_checkpoint is not None:
        command.extend(["--resume-from-checkpoint", str(resume_checkpoint)])
    if args.initial_model_checkpoint is not None:
        command.extend(
            ["--initial-model-checkpoint", str(args.initial_model_checkpoint)]
        )
    if args.save_initial_model_to is not None:
        command.extend(["--save-initial-model-to", str(args.save_initial_model_to)])
    command.extend(chunk_prompt_arg(prompt) for prompt in prompts)
    return command


def expected_validation_steps(leg: CurriculumLeg) -> tuple[int, ...]:
    start = 1 if leg.expected_resume_step is None else leg.expected_resume_step + 1
    regular = range(((start + 9) // 10) * 10, leg.max_steps + 1, 10)
    return tuple(sorted(set(regular) | set(leg.transition_steps)))


def main() -> None:
    args = parse_args()
    leg = LEGS[args.leg]
    if git_output("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError(f"Training must run on {EXPECTED_BRANCH}")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.run_prefix) is None:
        raise ValueError("invalid --run-prefix")
    for path in (args.packed_data_dir, args.packed_validation_data_dir):
        if not (path / "mix_manifest.json").is_file():
            raise FileNotFoundError(path)
    if args.initial_model_checkpoint is not None and args.resume_run_dir is not None:
        raise ValueError("initial checkpoint and resume run are mutually exclusive")

    resume_checkpoint = validate_resume(args, leg)
    probe, prompts = load_probe_prompts(args.probe)
    commit = git_output("rev-parse", "HEAD")
    seeds = seed_factors(args)
    run_name = (
        f"{args.run_prefix}_{leg.name}_{args.main}_"
        f"i{seeds['model_init_seed']}_d{seeds['data_order_seed']}_"
        f"r{seeds['train_runtime_seed']}_ctx{leg.seq_len}_"
        f"step{leg.max_steps}_{commit[:7]}"
    )
    run_dir = args.work_root / "runs" / run_name
    archive_dir = args.archive_root / run_name
    if run_dir.exists() or archive_dir.exists():
        raise FileExistsError(f"refusing to overwrite {run_name}")
    run_dir.mkdir(parents=True)

    command = build_train_command(
        args, leg, run_dir, resume_checkpoint, prompts
    )
    metadata = {
        "version": 1,
        "name": run_name,
        "commit": commit,
        "main": args.main,
        "leg": leg.name,
        "schedule": leg.schedule,
        "phase": leg.phase,
        "seq_len": leg.seq_len,
        "batch_size": leg.batch_size,
        "grad_accum_steps": 32,
        "bytes_per_update": BYTES_PER_UPDATE,
        "base_seq_len": BASE_SEQ_LEN,
        "max_steps": leg.max_steps,
        "expected_resume_step": leg.expected_resume_step,
        "transition_steps": list(leg.transition_steps),
        "total_train_bytes": TOTAL_TRAIN_BYTES,
        "seed_factors": seeds,
        "probe": str(args.probe),
        "probe_sha256": sha256_file(args.probe),
        "probe_record_count": len(probe["records"]),
        "packed_data_manifest_sha256": sha256_file(
            args.packed_data_dir / "mix_manifest.json"
        ),
        "packed_validation_manifest_sha256": sha256_file(
            args.packed_validation_data_dir / "mix_manifest.json"
        ),
        "resume_run_dir": (
            str(args.resume_run_dir) if args.resume_run_dir is not None else None
        ),
        "resume_checkpoint": (
            str(resume_checkpoint) if resume_checkpoint is not None else None
        ),
        "initial_model_checkpoint": (
            str(args.initial_model_checkpoint)
            if args.initial_model_checkpoint is not None
            else None
        ),
        "command": command,
    }
    (run_dir / "context_run_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    run_and_log(command, run_dir / "training_console.log")

    checkpoint, observed_step = latest_checkpoint(run_dir)
    if observed_step != leg.max_steps:
        raise RuntimeError(
            f"final checkpoint mismatch: expected={leg.max_steps} got={observed_step}"
        )
    chunk_dir = run_dir / "validation_chunks"
    observed_validation = tuple(
        sorted(
            int(path.stem.split("_")[-1])
            for path in chunk_dir.glob("chunks_step_*.json")
        )
    )
    expected_validation = expected_validation_steps(leg)
    if observed_validation != expected_validation:
        raise RuntimeError(
            "validation step mismatch: "
            f"expected={expected_validation} got={observed_validation}"
        )

    raw_dir = run_dir / "dense_raw"
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_dense_linguistic_chunks.py",
            "--probe",
            str(args.probe),
            "--chunk-report-dir",
            str(chunk_dir),
            "--training-metrics",
            str(run_dir / "training_metrics.csv"),
            "--model-name",
            args.main,
            "--seed",
            str(args.seed),
            "--model-init-seed",
            str(args.model_init_seed),
            "--data-order-seed",
            str(args.data_order_seed),
            "--train-runtime-seed",
            str(args.train_runtime_seed),
            "--output-dir",
            str(raw_dir),
        ],
        check=True,
    )
    summary_command = [
        sys.executable,
        "scripts/summarize_linguistic_boundary_screening.py",
    ]
    for path in sorted(raw_dir.glob("*.json")):
        summary_command.extend(["--input", str(path)])
    summary_command.extend(["--output-dir", str(run_dir / "dense_summary")])
    subprocess.run(summary_command, check=True)

    archive_run(run_dir, archive_dir, include_checkpoint=True)
    print(
        json.dumps(
            {"run": run_name, "archive": str(archive_dir), "checkpoint": str(checkpoint)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
