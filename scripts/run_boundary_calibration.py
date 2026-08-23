from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


MODEL_CONFIGS = {
    "t26": "configs/hnet_2stage_200m.json",
    "m3t1": "configs/hnet_2stage_200m_m3t1.json",
    "k1t1": "configs/hnet_2stage_200m_k1t1.json",
    "k1g1": "configs/hnet_2stage_200m_k1g1.json",
    "k3t1": "configs/hnet_2stage_200m_k3t1.json",
    "k3g1": "configs/hnet_2stage_200m_k3g1.json",
    "k1first_mix": "configs/hnet_2stage_200m_k1first_mix.json",
    "k3first_mix": "configs/hnet_2stage_200m_k3first_mix.json",
    "k14g12_front": "configs/hnet_2stage_200m_k14g12_front.json",
    "k14g12_middle": "configs/hnet_2stage_200m_k14g12_middle.json",
    "k14g12_late": "configs/hnet_2stage_200m_k14g12_late.json",
    "k15g11_split": "configs/hnet_2stage_200m_k15g11_split.json",
    "k16g10_even": "configs/hnet_2stage_200m_k16g10_even.json",
}
PROBES = [
    "彼女はいま笑っている。",
    "彼は失敗をごまかすように笑っている。",
    "高性能計算機資源を利用した。",
    "猫が机の下で眠っている。",
    "validation_lossをstep=55で確認する。",
    '{"tool":"search","query":"東京の天気"}',
]
ARCHIVE_PREFIX = Path("/content/drive/MyDrive/hnet_agent_200m_main")
EXPECTED_BRANCH = "200m_main"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        files.append({"path": relative, "bytes": size, "sha256": digest})
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return {
        "path": str(root),
        "files": files,
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "tree_sha256": combined.hexdigest(),
    }


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def ratio_tag(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def run_name(
    main_network: str,
    ratio_weight: float,
    commit: str,
    inner_compression_target: float = 3.0,
    outer_compression_target: float = 3.0,
    seed: int = 42,
) -> str:
    return (
        f"r5_cal_{main_network}_comp{ratio_tag(inner_compression_target)}-"
        f"{ratio_tag(outer_compression_target)}_"
        f"rw{ratio_tag(ratio_weight)}_"
        f"utf8hard_s{seed}_step55_{commit[:7]}"
    )


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in (
        "torch",
        "triton",
        "mamba-ssm",
        "flash-attn",
        "causal-conv1d",
        "fla-core",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def run_and_log(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def archive_run(run_dir: Path, archive_dir: Path, include_checkpoint: bool) -> None:
    if archive_dir.exists():
        raise FileExistsError(f"Archive already exists: {archive_dir}")
    archive_dir.mkdir(parents=True)
    for source in run_dir.iterdir():
        if source.name.startswith("checkpoint_step_") and not include_checkpoint:
            continue
        destination = archive_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one matched-condition 55-step boundary calibration."
    )
    parser.add_argument("--main", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--ratio-weight", type=float, choices=[0.03, 0.05, 0.08], required=True)
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], default=42)
    parser.add_argument(
        "--inner-compression-target",
        type=float,
        choices=[2.5, 3.0, 3.5],
        default=3.0,
        help="Compression target for the inner (L0-to-L1) stage.",
    )
    parser.add_argument(
        "--outer-compression-target",
        type=float,
        choices=[2.5, 3.0, 3.5],
        default=3.0,
        help="Compression target for the outer (L1-to-L2) stage.",
    )
    parser.add_argument("--packed-data-dir", type=Path, required=True)
    parser.add_argument("--packed-validation-data-dir", type=Path, required=True)
    parser.add_argument(
        "--work-root", type=Path, default=Path("/content/hnet_agent_200m_main_work")
    )
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_PREFIX / "runs")
    parser.add_argument("--archive-checkpoint", action="store_true")
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
    if git_output("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError(f"Calibration must run on {EXPECTED_BRANCH}")

    commit = git_output("rev-parse", "HEAD")
    name = run_name(
        args.main,
        args.ratio_weight,
        commit,
        seed=args.seed,
        inner_compression_target=args.inner_compression_target,
        outer_compression_target=args.outer_compression_target,
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
        dataset_manifest_path = args.work_root / "dataset_manifest.json"
        dataset_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_manifest_path.write_text(
            json.dumps(datasets, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

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
        "55",
        "--lr-schedule-steps",
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
        "command": command,
        "archive_checkpoint": args.archive_checkpoint,
    }
    (run_dir / "config.yaml").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    run_and_log(command, run_dir / "training_console.log")

    checkpoints = sorted(run_dir.glob("checkpoint_step_*.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected one final checkpoint, found {len(checkpoints)}")
    manifest = {
        "run_name": name,
        "commit": commit,
        "branch": EXPECTED_BRANCH,
        "git_dirty": bool(git_output("status", "--porcelain")),
        "model_config": MODEL_CONFIGS[args.main],
        "model_config_sha256": sha256_file(Path(MODEL_CONFIGS[args.main])),
        "datasets": datasets,
        "checkpoint": {
            "path": str(checkpoints[0]),
            "bytes": checkpoints[0].stat().st_size,
            "sha256": sha256_file(checkpoints[0]),
            "archived": args.archive_checkpoint,
        },
        "environment": {
            "python": sys.version,
            "packages": package_versions(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi": subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
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
    archive_run(run_dir, archive_dir, include_checkpoint=args.archive_checkpoint)
    print(json.dumps({"run": name, "archive": str(archive_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
