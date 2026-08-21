from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
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
    ratio_tag,
    run_and_log,
)


CHECKPOINT_STEPS = (55, 110, 165, 220)
DENSE_STEPS = tuple(range(10, 221, 10))
LR_SCHEDULE_STEPS = 220
OUTER_COMPRESSION_TARGETS = {
    "k1g1": 2.5,
    "k3g1": 2.5,
    "k3t1": 3.0,
    "k1first_mix": 2.5,
    "k3first_mix": 2.5,
}


def checkpoint_steps(max_steps: int) -> tuple[int, ...]:
    return tuple(step for step in CHECKPOINT_STEPS if step <= max_steps)


def dense_steps(max_steps: int) -> tuple[int, ...]:
    return tuple(range(10, max_steps + 1, 10))


def run_name(
    main_network: str, seed: int, commit: str, max_steps: int = 220
) -> str:
    return (
        f"r6_dense_family_v1_{main_network}_s{seed}_"
        f"step{max_steps}_{commit[:7]}"
    )


def copy_resume_artifacts(source: Path, destination: Path) -> Path:
    checkpoint = source / "checkpoint_step_000055.pt"
    chunks = source / "validation_chunks"
    required_files = (
        checkpoint,
        source / "training_metrics.csv",
        source / "validation_metrics.csv",
    )
    if not all(path.is_file() for path in required_files) or not chunks.is_dir():
        raise FileNotFoundError(f"Incomplete dense resume source: {source}")
    observed = tuple(
        int(path.stem.split("_")[-1])
        for path in sorted(chunks.glob("chunks_step_*.json"))
    )
    if observed != dense_steps(55):
        raise RuntimeError(f"Resume source dense steps are incomplete: {observed}")
    destination.mkdir(parents=True)
    for path in required_files:
        shutil.copy2(path, destination / path.name)
    shutil.copytree(chunks, destination / "validation_chunks")
    return destination / checkpoint.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense family-boundary training.")
    parser.add_argument(
        "--main", choices=sorted(OUTER_COMPRESSION_TARGETS), required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, choices=[55, 220], default=220)
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        help="Completed 55-step dense run whose optimizer/data state is resumed.",
    )
    parser.add_argument("--packed-data-dir", type=Path, required=True)
    parser.add_argument("--packed-validation-data-dir", type=Path, required=True)
    parser.add_argument(
        "--probe",
        type=Path,
        default=Path("configs/linguistic_boundary_family_probe_v1.json"),
    )
    parser.add_argument(
        "--work-root", type=Path, default=Path("/content/hnet_agent_200m_main_work")
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=ARCHIVE_PREFIX / "runs/dense_family_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if git_output("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError(f"Training must run on {EXPECTED_BRANCH}")
    for path in (args.packed_data_dir, args.packed_validation_data_dir):
        if not path.is_dir() or not (path / "mix_manifest.json").is_file():
            raise FileNotFoundError(path)
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    prompts = [record["text"] for record in probe["records"]]
    if len(prompts) != 24 or len(set(prompts)) != len(prompts):
        raise ValueError("dense family probe must contain 24 unique texts")

    commit = git_output("rev-parse", "HEAD")
    name = run_name(args.main, args.seed, commit, args.max_steps)
    run_dir = args.work_root / "runs" / name
    archive_dir = args.archive_root / name
    if run_dir.exists() or archive_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {name}")
    if args.resume_run_dir is not None:
        if args.max_steps != 220:
            raise ValueError("--resume-run-dir requires --max-steps 220")
        resume_checkpoint = copy_resume_artifacts(args.resume_run_dir, run_dir)
    else:
        run_dir.mkdir(parents=True)
        resume_checkpoint = None

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
        str(args.max_steps),
        "--lr-schedule-steps",
        str(LR_SCHEDULE_STEPS),
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
        "--num-workers",
        "0",
    ]
    if resume_checkpoint is not None:
        command.extend(["--resume-from-checkpoint", str(resume_checkpoint)])
    for prompt in prompts:
        command.extend(["--chunk-prompt", prompt])
    (run_dir / "dense_run_config.json").write_text(
        json.dumps(
            {
                "name": name,
                "commit": commit,
                "main": args.main,
                "seed": args.seed,
                "probe": str(args.probe),
                "resume_run_dir": (
                    str(args.resume_run_dir) if args.resume_run_dir is not None else None
                ),
                "command": command,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    run_and_log(command, run_dir / "training_console.log")

    checkpoints = sorted(run_dir.glob("checkpoint_step_*.pt"))
    if [path.name for path in checkpoints] != [
        f"checkpoint_step_{step:06d}.pt" for step in checkpoint_steps(args.max_steps)
    ]:
        raise RuntimeError("dense run checkpoint set is incomplete")
    chunk_reports = sorted((run_dir / "validation_chunks").glob("chunks_step_*.json"))
    observed_steps = tuple(int(path.stem.split("_")[-1]) for path in chunk_reports)
    if observed_steps != dense_steps(args.max_steps):
        raise RuntimeError(f"dense chunk steps are incomplete: {observed_steps}")

    raw_dir = run_dir / "dense_raw"
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_dense_linguistic_chunks.py",
            "--probe",
            str(args.probe),
            "--chunk-report-dir",
            str(run_dir / "validation_chunks"),
            "--training-metrics",
            str(run_dir / "training_metrics.csv"),
            "--model-name",
            args.main,
            "--seed",
            str(args.seed),
            "--output-dir",
            str(raw_dir),
        ],
        check=True,
    )
    summary_command = [sys.executable, "scripts/summarize_linguistic_boundary_screening.py"]
    for path in sorted(raw_dir.glob("*.json")):
        summary_command.extend(["--input", str(path)])
    summary_command.extend(["--output-dir", str(run_dir / "dense_summary")])
    subprocess.run(summary_command, check=True)
    archive_run(run_dir, archive_dir, include_checkpoint=True)
    print(json.dumps({"run": name, "archive": str(archive_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
