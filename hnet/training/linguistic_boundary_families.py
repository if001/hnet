from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable

@dataclass(frozen=True)
class FamilyLandmark:
    name: str
    segmentation: str


def landmark_byte_offset(surface: str, landmark: FamilyLandmark) -> int:
    """Return the sole boundary offset encoded by a family landmark."""
    if landmark.segmentation.replace("|", "") != surface:
        raise ValueError(
            "family landmark segmentation does not reconstruct its surface"
        )
    pieces = landmark.segmentation.split("|")
    if any(not piece for piece in pieces):
        raise ValueError("family landmark segmentation contains an empty piece")
    offsets: set[int] = set()
    prefix = ""
    for piece in pieces[:-1]:
        prefix += piece
        offsets.add(len(prefix.encode("utf-8")))
    if len(offsets) != 1:
        raise ValueError(
            "family landmark segmentation must contain exactly one boundary"
        )
    return next(iter(offsets))


def validate_family_annotation(record: dict[str, Any]) -> None:
    family = record.get("family")
    if family is None:
        return
    if not family.get("group"):
        raise ValueError("family group must be non-empty")
    if family.get("kind") not in {"paradigm", "lexical", "context_control"}:
        raise ValueError(f"unsupported family kind: {family.get('kind')!r}")
    landmarks = family.get("landmarks")
    if not landmarks:
        raise ValueError("family annotation must contain landmarks")
    names: set[str] = set()
    surface = record["focus"]["surface"]
    for item in landmarks:
        landmark = FamilyLandmark(item["name"], item["segmentation"])
        if not landmark.name or landmark.name in names:
            raise ValueError("family landmark names must be non-empty and unique")
        names.add(landmark.name)
        landmark_byte_offset(surface, landmark)


def _optional_mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return mean(present) if present else None


def summarize_family_scores(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize family-level quality and named-landmark selection.

    Consistency is intentionally reported separately from coverage: selecting no
    landmark in every record is perfectly consistent but provides zero coverage.
    """
    family_groups: dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    landmark_groups: dict[
        tuple[str, str, str, str, str], list[bool]
    ] = defaultdict(list)
    metadata: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for run in runs:
        run_id = f"{run['model_name']}|{run.get('seed')}|{run.get('checkpoint_label')}"
        for record in run["records"]:
            family = record.get("family")
            if family is None:
                continue
            validate_family_annotation(record)
            surface = record["focus"]["surface"]
            for condition, result in record["conditions"].items():
                for stage in ("stage0", "stage1"):
                    key = (run_id, condition, stage, family["group"])
                    score = result[stage]["score"]
                    family_groups[key].append(
                        {
                            "record_id": record["id"],
                            "surface": surface,
                            "kind": family["kind"],
                            "score": score,
                        }
                    )
                    metadata[key] = {
                        "model_name": run["model_name"],
                        "seed": run.get("seed"),
                        "checkpoint_label": run.get("checkpoint_label"),
                    }
                    selected = set(score["selected_evaluable_offsets"])
                    for item in family["landmarks"]:
                        landmark = FamilyLandmark(item["name"], item["segmentation"])
                        offset = landmark_byte_offset(surface, landmark)
                        landmark_groups[(*key, landmark.name)].append(
                            offset in selected
                        )

    family_rows: list[dict[str, Any]] = []
    for key, records in sorted(family_groups.items()):
        run_id, condition, stage, family_group = key
        scores = [record["score"] for record in records]
        selected = sum(len(score["selected_evaluable_offsets"]) for score in scores)
        explained = sum(len(score["explained_offsets"]) for score in scores)
        unexplained = sum(len(score["unexplained_offsets"]) for score in scores)
        acceptable = sum(len(score["acceptable_offsets"]) for score in scores)
        fractures = sum(
            len(score.get("lexeme_fracture_offsets", ())) for score in scores
        )
        signatures = [tuple(score["selected_evaluable_offsets"]) for score in scores]
        signature_mode = Counter(signatures).most_common(1)[0][1]
        kind = records[0]["kind"]
        same_surface = len({record["surface"] for record in records}) == 1
        context_consistency = (
            signature_mode / len(records)
            if kind == "context_control" and same_surface
            else None
        )
        family_rows.append(
            {
                **metadata[key],
                "run_id": run_id,
                "condition": condition,
                "stage": stage,
                "family_group": family_group,
                "family_kind": kind,
                "records": len(records),
                "explainable_boundary_precision": explained / selected
                if selected
                else None,
                "category_coverage": explained / acceptable if acceptable else None,
                "unexplained_boundary_rate": unexplained / selected
                if selected
                else None,
                "family_lexeme_integrity": sum(
                    not score.get("lexeme_fracture_offsets") for score in scores
                )
                / len(scores),
                "lexeme_fracture_rate": fractures / selected if selected else None,
                "no_unexplained_record_rate": sum(
                    not score["unexplained_offsets"] for score in scores
                )
                / len(scores),
                "context_signature_consistency": context_consistency,
                "best_segmentation_f1_mean": _optional_mean(
                    score.get("best_segmentation", {}).get("f1") for score in scores
                ),
            }
        )

    landmark_rows: list[dict[str, Any]] = []
    for key, selections in sorted(landmark_groups.items()):
        run_id, condition, stage, family_group, landmark_name = key
        selected_count = sum(selections)
        majority_count = max(selected_count, len(selections) - selected_count)
        landmark_rows.append(
            {
                **metadata[(run_id, condition, stage, family_group)],
                "run_id": run_id,
                "condition": condition,
                "stage": stage,
                "family_group": family_group,
                "landmark_name": landmark_name,
                "records": len(selections),
                "landmark_coverage": selected_count / len(selections),
                "landmark_consistency": majority_count / len(selections),
                "all_selected": selected_count == len(selections),
                "none_selected": selected_count == 0,
            }
        )
    return family_rows, landmark_rows
