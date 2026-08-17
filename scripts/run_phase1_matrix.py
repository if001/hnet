from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CONDITIONS = {
    "t26": {"ratio_weight": 0.08, "inner": 3.0, "outer": 2.5},
    "k1t1": {"ratio_weight": 0.05, "inner": 3.0, "outer": 3.0},
    "m3t1": {"ratio_weight": 0.05, "inner": 3.0, "outer": 2.5},
    "k1g1": {"ratio_weight": 0.08, "inner": 3.0, "outer": 2.5},
    "k3t1": {"ratio_weight": 0.05, "inner": 3.0, "outer": 3.0},
    "k3g1": {"ratio_weight": 0.08, "inner": 3.0, "outer": 2.5},
}


def parse_run(value: str) -> tuple[str, int]:
    try:
        main, raw_seed = value.split(":", maxsplit=1)
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("run must use MAIN:SEED") from exc
    if main not in CONDITIONS:
        raise argparse.ArgumentTypeError(f"unknown main network: {main}")
    if seed not in {42, 43, 44}:
        raise argparse.ArgumentTypeError("seed must be 42, 43, or 44")
    return main, seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the missing Phase 1 matrix.")
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--packed-data-dir", type=Path, required=True)
    parser.add_argument("--packed-validation-data-dir", type=Path, required=True)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/hnet_agent_200m_main/runs/phase1_calibration"
        ),
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/hnet_agent_200m_main/manifests/phase1_matrix_status.json"
        ),
    )
    return parser.parse_args()


def write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    cu13_lib = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib"
    env["LD_LIBRARY_PATH"] = f"{cu13_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    status: dict[str, object] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "requested_runs": [f"{main}:{seed}" for main, seed in args.run],
        "runs": [],
        "state": "running",
    }
    write_status(args.status_path, status)

    run_statuses: list[dict[str, object]] = []
    for main_network, seed in args.run:
        condition = CONDITIONS[main_network]
        command = [
            sys.executable,
            "scripts/run_matched_boundary_training.py",
            "--main",
            main_network,
            "--ratio-weight",
            str(condition["ratio_weight"]),
            "--inner-compression-target",
            str(condition["inner"]),
            "--outer-compression-target",
            str(condition["outer"]),
            "--seed",
            str(seed),
            "--packed-data-dir",
            str(args.packed_data_dir),
            "--packed-validation-data-dir",
            str(args.packed_validation_data_dir),
            "--archive-root",
            str(args.archive_root / main_network),
        ]
        item: dict[str, object] = {
            "main_network": main_network,
            "seed": seed,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "state": "running",
        }
        run_statuses.append(item)
        status["runs"] = run_statuses
        write_status(args.status_path, status)
        try:
            subprocess.run(command, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            item["state"] = "failed"
            item["return_code"] = exc.returncode
            item["finished_at"] = datetime.now(timezone.utc).isoformat()
            status["state"] = "failed"
            status["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_status(args.status_path, status)
            raise
        item["state"] = "completed"
        item["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_status(args.status_path, status)

    status["state"] = "completed"
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_status(args.status_path, status)


if __name__ == "__main__":
    main()
