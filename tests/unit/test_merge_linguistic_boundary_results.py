from copy import deepcopy

import pytest

from scripts.merge_linguistic_boundary_results import (
    checkpoint_step,
    merge_payloads,
    result_key,
)


def _payload(label: str, record_id: str) -> dict:
    return {
        "version": 3,
        "model_name": "k3g1",
        "model_path": "/checkpoints/model.pt",
        "config_path": "/configs/model.json",
        "seed": 42,
        "checkpoint_label": label,
        "probe_path": "/probe.json",
        "byte_boundary_constraint": "utf8-hard",
        "byte_boundary_constraint_bias": 0.0,
        "budget_profiles": [{"id": "central"}],
        "records": [{"id": record_id}],
    }


def test_checkpoint_step_ignores_seed_number() -> None:
    assert checkpoint_step("seed42-step_000220-family-v1") == 220
    assert result_key(_payload("step55-v2", "category")) == ("k3g1", 42, 55)


def test_merge_payloads_combines_disjoint_records() -> None:
    category = _payload("step55-v2", "category")
    family = _payload("step55-family-v1", "family")
    merged = merge_payloads(
        category,
        family,
        expected_category_records=1,
        expected_family_records=1,
    )
    assert merged["checkpoint_label"] == "step55-combined112-v1"
    assert merged["record_sets"] == {"category": 1, "family": 1, "total": 2}
    assert [record["id"] for record in merged["records"]] == [
        "category",
        "family",
    ]


def test_merge_payloads_rejects_metadata_mismatch() -> None:
    category = _payload("step55-v2", "category")
    family = deepcopy(_payload("step55-family-v1", "family"))
    family["model_path"] = "/checkpoints/different.pt"
    with pytest.raises(ValueError, match="metadata mismatch"):
        merge_payloads(
            category,
            family,
            expected_category_records=1,
            expected_family_records=1,
        )


def test_merge_payloads_rejects_record_id_overlap() -> None:
    category = _payload("step55-v2", "same")
    family = _payload("step55-family-v1", "same")
    with pytest.raises(ValueError, match="overlap"):
        merge_payloads(
            category,
            family,
            expected_category_records=1,
            expected_family_records=1,
        )
