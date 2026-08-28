from __future__ import annotations

import argparse
import json
import re
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
    sha256_file,
)


CHECKPOINT_INTERVAL = 55
DENSE_STEPS = tuple(range(10, 221, 10))
LR_SCHEDULE_STEPS = 220
OUTER_COMPRESSION_TARGETS = {
    "k1g1": 2.5,
    "k3g1": 2.5,
    "k3t1": 3.0,
    "k1first_mix": 2.5,
    "k3first_mix": 2.5,
    "k14g12_front": 2.5,
    "k14g12_middle": 2.5,
    "k14g12_late": 2.5,
    "k15g11_split": 2.5,
    "k16g10_even": 2.5,
    "t26": 2.5,
}


def checkpoint_steps(max_steps: int) -> tuple[int, ...]:
    steps = list(range(CHECKPOINT_INTERVAL, max_steps + 1, CHECKPOINT_INTERVAL))
    if not steps or steps[-1] != max_steps:
        steps.append(max_steps)
    return tuple(steps)


def dense_steps(max_steps: int) -> tuple[int, ...]:
    return tuple(range(10, max_steps + 1, 10))


def run_name(
    main_network: str,
    seed: int,
    commit: str,
    max_steps: int = 220,
    *,
    model_init_seed: int | None = None,
    data_order_seed: int | None = None,
    train_runtime_seed: int | None = None,
    run_prefix: str = "r6_dense_family_v1",
    boundary_feature_mode: str = "final",
    boundary_feature_final_logit_bias: float = 2.0,
) -> str:
    if any(
        value is not None
        for value in (model_init_seed, data_order_seed, train_runtime_seed)
    ):
        init_seed = seed if model_init_seed is None else model_init_seed
        data_seed = seed if data_order_seed is None else data_order_seed
        runtime_seed = seed if train_runtime_seed is None else train_runtime_seed
        seed_tag = f"i{init_seed}_d{data_seed}_r{runtime_seed}"
    else:
        seed_tag = f"s{seed}"
    feature_tag = ""
    if boundary_feature_mode == "layer-scalar-mix":
        feature_tag = (
            f"_bfmix_fb{ratio_tag(boundary_feature_final_logit_bias)}"
        )
    elif boundary_feature_mode != "final":
        raise ValueError(
            f"Unsupported boundary feature mode: {boundary_feature_mode!r}"
        )
    return (
        f"{run_prefix}_{main_network}_{seed_tag}{feature_tag}_"
        f"step{max_steps}_{commit[:7]}"
    )


def load_probe_prompts(path: Path) -> tuple[dict[str, object], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("dense probe must contain at least one record")
    record_texts = [
        record.get("text") for record in records if isinstance(record, dict)
    ]
    if len(record_texts) != len(records) or not all(
        isinstance(prompt, str) and prompt.strip() for prompt in record_texts
    ):
        raise ValueError("every dense probe record must contain non-empty text")
    prompts = list(dict.fromkeys(record_texts))
    return payload, prompts


def chunk_prompt_arg(prompt: str) -> str:
    """Keep option-looking probe text attached to its argparse option."""
    return f"--chunk-prompt={prompt}"


def copy_resume_artifacts(source: Path, destination: Path) -> Path:
    checkpoints = sorted(source.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint in dense resume source: {source}")
    checkpoint = checkpoints[-1]
    source_step = int(checkpoint.stem.split("_")[-1])
    chunks = source / "validation_chunks"
    required_files = (
        source / "training_metrics.csv",
        source / "validation_metrics.csv",
    )
    if not all(path.is_file() for path in required_files) or not chunks.is_dir():
        raise FileNotFoundError(f"Incomplete dense resume source: {source}")
    observed = tuple(
        int(path.stem.split("_")[-1])
        for path in sorted(chunks.glob("chunks_step_*.json"))
    )
    if observed != dense_steps(source_step):
        raise RuntimeError(f"Resume source dense steps are incomplete: {observed}")
    destination.mkdir(parents=True)
    for path in required_files:
        shutil.copy2(path, destination / path.name)
    for path in checkpoints:
        shutil.copy2(path, destination / path.name)
    shutil.copytree(chunks, destination / "validation_chunks")
    return destination / checkpoint.name


def resume_seed_factors(source: Path) -> dict[str, int]:
    config_path = source / "dense_run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    factors = payload.get("seed_factors")
    if isinstance(factors, dict) and all(
        isinstance(factors.get(key), int)
        for key in ("model_init_seed", "data_order_seed", "train_runtime_seed")
    ):
        return {
            key: int(factors[key])
            for key in ("model_init_seed", "data_order_seed", "train_runtime_seed")
        }
    legacy_seed = payload.get("seed")
    if not isinstance(legacy_seed, int):
        raise ValueError(f"resume source has no valid seed metadata: {config_path}")
    return {
        "model_init_seed": legacy_seed,
        "data_order_seed": legacy_seed,
        "train_runtime_seed": legacy_seed,
    }


def resume_boundary_feature(source: Path) -> dict[str, object]:
    config_path = source / "dense_run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    feature = payload.get("boundary_feature")
    if not isinstance(feature, dict):
        return {"mode": "final", "final_logit_bias": 2.0}
    return {
        "mode": str(feature.get("mode", "final")),
        "final_logit_bias": float(feature.get("final_logit_bias", 2.0)),
    }


def resume_family_consistency(source: Path) -> dict[str, object]:
    payload = json.loads(
        (source / "dense_run_config.json").read_text(encoding="utf-8")
    )
    feature = payload.get("family_consistency")
    if not isinstance(feature, dict):
        return {
            "data_sha256": None,
            "split": "train",
            "objective": "c1",
            "weight": 0.0,
            "margin": 0.15,
            "seed": 42,
        }
    return {
        "data_sha256": feature.get("data_sha256"),
        "split": str(feature.get("split", "train")),
        "objective": str(feature.get("objective", "c1")),
        "weight": float(feature.get("weight", 0.0)),
        "margin": float(feature.get("margin", 0.15)),
        "seed": int(feature.get("seed", 42)),
    }


def resume_ffn_moe(source: Path) -> dict[str, object]:
    payload = json.loads(
        (source / "dense_run_config.json").read_text(encoding="utf-8")
    )
    feature = payload.get("ffn_moe")
    if not isinstance(feature, dict):
        return {"enabled": False, "aux_loss_weight": 0.0}
    return {
        "enabled": bool(feature.get("enabled", False)),
        "aux_loss_weight": float(feature.get("aux_loss_weight", 0.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense family-boundary training.")
    parser.add_argument(
        "--main", choices=sorted(OUTER_COMPRESSION_TARGETS), required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-init-seed", type=int)
    parser.add_argument("--data-order-seed", type=int)
    parser.add_argument("--train-runtime-seed", type=int)
    parser.add_argument(
        "--initial-model-checkpoint",
        type=Path,
        help="Exact step-0 state used for data-order-only comparisons.",
    )
    parser.add_argument(
        "--save-initial-model-to",
        type=Path,
        help="Optional path for the exact pre-training model state.",
    )
    parser.add_argument(
        "--max-steps", type=int, choices=[20, 55, 100, 200, 220], default=220
    )
    parser.add_argument(
        "--freeze-mode",
        choices=("none", "router", "main", "outer"),
        default="none",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        help=(
            "Completed 55-step dense run whose optimizer/data state is resumed "
            "to step 100 or 220."
        ),
    )
    parser.add_argument("--packed-data-dir", type=Path, required=True)
    parser.add_argument("--packed-validation-data-dir", type=Path, required=True)
    parser.add_argument(
        "--probe",
        type=Path,
        default=Path("configs/linguistic_boundary_family_probe_v1.json"),
    )
    parser.add_argument(
        "--run-prefix",
        default="r6_dense_family_v1",
        help="Artifact name prefix; use a distinct value for a new probe protocol.",
    )
    parser.add_argument(
        "--boundary-feature-mode",
        choices=["final", "layer-scalar-mix"],
        default="final",
    )
    parser.add_argument(
        "--boundary-feature-final-logit-bias",
        type=float,
        default=2.0,
    )
    parser.add_argument("--family-consistency-data", type=Path)
    parser.add_argument("--family-consistency-split", default="train")
    parser.add_argument(
        "--family-consistency-objective", choices=("c1", "c2"), default="c1"
    )
    parser.add_argument("--family-consistency-weight", type=float, default=0.0)
    parser.add_argument("--family-consistency-margin", type=float, default=0.15)
    parser.add_argument("--family-consistency-seed", type=int, default=42)
    parser.add_argument("--ffn-moe", action="store_true")
    parser.add_argument("--moe-aux-loss-weight", type=float, default=0.01)
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
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.run_prefix) is None:
        raise ValueError(
            "--run-prefix must contain only lowercase letters, digits, _ or -"
        )
    _probe, prompts = load_probe_prompts(args.probe)

    commit = git_output("rev-parse", "HEAD")
    resolved_seeds = {
        "model_init_seed": (
            args.seed if args.model_init_seed is None else args.model_init_seed
        ),
        "data_order_seed": (
            args.seed if args.data_order_seed is None else args.data_order_seed
        ),
        "train_runtime_seed": (
            args.seed if args.train_runtime_seed is None else args.train_runtime_seed
        ),
    }
    name = run_name(
        args.main,
        args.seed,
        commit,
        args.max_steps,
        model_init_seed=resolved_seeds["model_init_seed"],
        data_order_seed=resolved_seeds["data_order_seed"],
        train_runtime_seed=resolved_seeds["train_runtime_seed"],
        run_prefix=args.run_prefix,
        boundary_feature_mode=args.boundary_feature_mode,
        boundary_feature_final_logit_bias=(
            args.boundary_feature_final_logit_bias
        ),
    )
    run_dir = args.work_root / "runs" / name
    archive_dir = args.archive_root / name
    if run_dir.exists() or archive_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {name}")
    if args.resume_run_dir is not None:
        if args.max_steps not in {100, 220}:
            raise ValueError(
                "--resume-run-dir requires --max-steps 100 or 220"
            )
        source_seeds = resume_seed_factors(args.resume_run_dir)
        if source_seeds != resolved_seeds:
            raise ValueError(
                f"resume seed mismatch: source={source_seeds} requested={resolved_seeds}"
            )
        requested_feature = {
            "mode": args.boundary_feature_mode,
            "final_logit_bias": args.boundary_feature_final_logit_bias,
        }
        source_feature = resume_boundary_feature(args.resume_run_dir)
        if source_feature != requested_feature:
            raise ValueError(
                "resume boundary-feature mismatch: "
                f"source={source_feature} requested={requested_feature}"
            )
        requested_family = {
            "data_sha256": (
                sha256_file(args.family_consistency_data)
                if args.family_consistency_data is not None
                else None
            ),
            "split": args.family_consistency_split,
            "objective": args.family_consistency_objective,
            "weight": args.family_consistency_weight,
            "margin": args.family_consistency_margin,
            "seed": args.family_consistency_seed,
        }
        source_family = resume_family_consistency(args.resume_run_dir)
        if source_family != requested_family:
            raise ValueError(
                "resume family-consistency mismatch: "
                f"source={source_family} requested={requested_family}"
            )
        requested_moe = {
            "enabled": args.ffn_moe,
            "aux_loss_weight": args.moe_aux_loss_weight if args.ffn_moe else 0.0,
        }
        source_moe = resume_ffn_moe(args.resume_run_dir)
        if source_moe != requested_moe:
            raise ValueError(
                f"resume FFN-MoE mismatch: source={source_moe} "
                f"requested={requested_moe}"
            )
        resume_checkpoint = copy_resume_artifacts(args.resume_run_dir, run_dir)
        resume_source_steps = tuple(
            int(path.stem.split("_")[-1])
            for path in sorted(args.resume_run_dir.glob("checkpoint_step_*.pt"))
        )
    else:
        run_dir.mkdir(parents=True)
        resume_checkpoint = None
        resume_source_steps = ()
    if args.initial_model_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError(
            "--initial-model-checkpoint cannot be combined with --resume-run-dir"
        )

    base_model_config_path = Path(MODEL_CONFIGS[args.main])
    model_config_path = base_model_config_path
    if args.boundary_feature_mode == "layer-scalar-mix" or args.ffn_moe:
        model_config_payload = json.loads(
            base_model_config_path.read_text(encoding="utf-8")
        )
        if args.boundary_feature_mode == "layer-scalar-mix":
            model_config_payload["boundary_feature_cfg"] = {
                "mode": "layer_scalar_mix",
                "final_logit_bias": args.boundary_feature_final_logit_bias,
            }
        if args.ffn_moe:
            model_config_payload["ffn_moe_cfg"] = {
                "enabled": True,
                "num_experts": 4,
                "top_k": 1,
                "capacity_factor": 1.25,
            }
        model_config_path = run_dir / "input_model_config.json"
        model_config_path.write_text(
            json.dumps(model_config_payload, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )

    command = [
        sys.executable,
        "train.py",
        "--model-config-path",
        str(model_config_path),
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
        "--model-init-seed",
        str(resolved_seeds["model_init_seed"]),
        "--data-order-seed",
        str(resolved_seeds["data_order_seed"]),
        "--train-runtime-seed",
        str(resolved_seeds["train_runtime_seed"]),
        "--num-workers",
        "0",
        "--freeze-mode",
        args.freeze_mode,
    ]
    if resume_checkpoint is not None:
        command.extend(["--resume-from-checkpoint", str(resume_checkpoint)])
    if args.initial_model_checkpoint is not None:
        command.extend(
            ["--initial-model-checkpoint", str(args.initial_model_checkpoint)]
        )
    if args.save_initial_model_to is not None:
        command.extend(["--save-initial-model-to", str(args.save_initial_model_to)])
    if args.family_consistency_data is not None:
        command.extend(
            [
                "--family-consistency-data",
                str(args.family_consistency_data),
                "--family-consistency-split",
                args.family_consistency_split,
                "--family-consistency-objective",
                args.family_consistency_objective,
                "--family-consistency-weight",
                str(args.family_consistency_weight),
                "--family-consistency-margin",
                str(args.family_consistency_margin),
                "--family-consistency-seed",
                str(args.family_consistency_seed),
            ]
        )
    if args.ffn_moe:
        command.extend(
            ["--moe-aux-loss-weight", str(args.moe_aux_loss_weight)]
        )
    for prompt in prompts:
        command.append(chunk_prompt_arg(prompt))
    (run_dir / "dense_run_config.json").write_text(
        json.dumps(
            {
                "name": name,
                "commit": commit,
                "main": args.main,
                "seed": args.seed,
                "seed_factors": resolved_seeds,
                "probe": str(args.probe),
                "probe_record_count": len(_probe["records"]),
                "probe_unique_prompt_count": len(prompts),
                "probe_sha256": sha256_file(args.probe),
                "run_prefix": args.run_prefix,
                "freeze_mode": args.freeze_mode,
                "boundary_feature": {
                    "mode": args.boundary_feature_mode,
                    "final_logit_bias": args.boundary_feature_final_logit_bias,
                },
                "family_consistency": {
                    "data": (
                        str(args.family_consistency_data)
                        if args.family_consistency_data is not None
                        else None
                    ),
                    "data_sha256": (
                        sha256_file(args.family_consistency_data)
                        if args.family_consistency_data is not None
                        else None
                    ),
                    "split": args.family_consistency_split,
                    "objective": args.family_consistency_objective,
                    "weight": args.family_consistency_weight,
                    "margin": args.family_consistency_margin,
                    "seed": args.family_consistency_seed,
                },
                "ffn_moe": {
                    "enabled": args.ffn_moe,
                    "aux_loss_weight": (
                        args.moe_aux_loss_weight if args.ffn_moe else 0.0
                    ),
                    "num_experts": 4 if args.ffn_moe else 0,
                    "top_k": 1 if args.ffn_moe else 0,
                    "capacity_factor": 1.25 if args.ffn_moe else 0.0,
                },
                "base_model_config": str(base_model_config_path),
                "base_model_config_sha256": sha256_file(base_model_config_path),
                "model_config_sha256": sha256_file(model_config_path),
                "packed_data_manifest_sha256": sha256_file(
                    args.packed_data_dir / "mix_manifest.json"
                ),
                "packed_validation_manifest_sha256": sha256_file(
                    args.packed_validation_data_dir / "mix_manifest.json"
                ),
                "resume_run_dir": (
                    str(args.resume_run_dir) if args.resume_run_dir is not None else None
                ),
                "initial_model_checkpoint": (
                    str(args.initial_model_checkpoint)
                    if args.initial_model_checkpoint is not None
                    else None
                ),
                "save_initial_model_to": (
                    str(args.save_initial_model_to)
                    if args.save_initial_model_to is not None
                    else None
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
    expected_checkpoint_steps = tuple(
        sorted(set(checkpoint_steps(args.max_steps)) | set(resume_source_steps))
    )
    if [path.name for path in checkpoints] != [
        f"checkpoint_step_{step:06d}.pt" for step in expected_checkpoint_steps
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
            "--model-init-seed",
            str(resolved_seeds["model_init_seed"]),
            "--data-order-seed",
            str(resolved_seeds["data_order_seed"]),
            "--train-runtime-seed",
            str(resolved_seeds["train_runtime_seed"]),
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
