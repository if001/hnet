from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hnet.training.linguistic_boundaries import FocusAnnotation, score_focus_boundaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert in-training chunk JSON reports to probe result JSON."
    )
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--chunk-report-dir", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_cumulative_bytes(path: Path) -> dict[int, int]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        int(row["step"]): int(row["cumulative_input_bytes"])
        for row in rows
    }


def main() -> None:
    args = parse_args()
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    records_by_text = {record["text"]: record for record in probe["records"]}
    if len(records_by_text) != len(probe["records"]):
        raise ValueError("dense probe record texts must be unique")
    cumulative_bytes = load_cumulative_bytes(args.training_metrics)
    reports = sorted(args.chunk_report_dir.glob("chunks_step_*.json"))
    if not reports:
        raise FileNotFoundError(f"no dense chunk reports in {args.chunk_report_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        step = int(report["step"])
        report_by_text = {record["text"]: record for record in report["records"]}
        if set(report_by_text) != set(records_by_text):
            raise ValueError(f"probe/report text mismatch: {report_path}")
        output_records: list[dict[str, Any]] = []
        for text, annotation_record in records_by_text.items():
            observed = report_by_text[text]
            focus = annotation_record["focus"]
            annotation = FocusAnnotation(
                surface=focus["surface"],
                occurrence=int(focus.get("occurrence", 0)),
                acceptable_segmentations=tuple(focus["acceptable_segmentations"]),
                protected_substrings=tuple(focus.get("protected_substrings", ())),
            )
            conditions: dict[str, Any] = {"native": {}}
            for stage in ("stage0", "stage1"):
                positions = observed[stage]["boundary_positions"]
                conditions["native"][stage] = {
                    "boundary_positions": positions,
                    "boundary_probability": observed[stage][
                        "boundary_probability"
                    ],
                    "score": score_focus_boundaries(text, annotation, positions),
                }
            output_records.append(
                {
                    "id": annotation_record["id"],
                    "category": annotation_record["category"],
                    "text": text,
                    "focus": focus,
                    "pair": annotation_record.get("pair"),
                    "family": annotation_record.get("family"),
                    "conditions": conditions,
                }
            )
        output = {
            "version": int(probe["version"]),
            "model_name": args.model_name,
            "seed": args.seed,
            "checkpoint_label": f"step{step}-dense-native-v1",
            "cumulative_input_bytes": cumulative_bytes[step],
            "probe_path": str(args.probe),
            "records": output_records,
        }
        output_path = args.output_dir / (
            f"{args.model_name}_s{args.seed}_step{step}_dense.json"
        )
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(args.output_dir)


if __name__ == "__main__":
    main()
