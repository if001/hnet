import json

from scripts.run_dense_linguistic_training import (
    DENSE_STEPS,
    LR_SCHEDULE_STEPS,
    chunk_prompt_arg,
    OUTER_COMPRESSION_TARGETS,
    checkpoint_steps,
    copy_resume_artifacts,
    dense_steps,
    load_probe_prompts,
    resume_boundary_feature,
    resume_seed_factors,
    run_name,
)


def test_chunk_prompt_arg_preserves_option_looking_text() -> None:
    assert chunk_prompt_arg("--learning-rateを指定する。") == (
        "--chunk-prompt=--learning-rateを指定する。"
    )


def test_dense_run_name_records_main_seed_and_commit() -> None:
    assert run_name("k1g1", 42, "abcdef0123") == (
        "r6_dense_family_v1_k1g1_s42_step220_abcdef0"
    )


def test_dense_steps_are_ten_step_intervals() -> None:
    assert DENSE_STEPS == tuple(range(10, 221, 10))
    assert LR_SCHEDULE_STEPS == 220
    assert dense_steps(55) == (10, 20, 30, 40, 50)
    assert checkpoint_steps(55) == (55,)
    assert dense_steps(100) == tuple(range(10, 101, 10))
    assert checkpoint_steps(100) == (55, 100)
    assert checkpoint_steps(220) == (55, 110, 165, 220)


def test_dense_targets_include_combined_pareto_candidates() -> None:
    assert OUTER_COMPRESSION_TARGETS == {
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


def test_dense_run_name_supports_screening_length() -> None:
    assert run_name("k1first_mix", 42, "abcdef0123", 55) == (
        "r6_dense_family_v1_k1first_mix_s42_step55_abcdef0"
    )


def test_dense_run_name_records_boundary_feature_variant() -> None:
    assert run_name(
        "t26",
        42,
        "abcdef0123",
        100,
        model_init_seed=42,
        data_order_seed=42,
        train_runtime_seed=42,
        run_prefix="r9_boundary_fusion_v1",
        boundary_feature_mode="layer-scalar-mix",
        boundary_feature_final_logit_bias=2.0,
    ) == "r9_boundary_fusion_v1_t26_i42_d42_r42_bfmix_fb2_step100_abcdef0"


def test_dense_run_name_records_factorized_seeds() -> None:
    assert run_name(
        "k14g12_late",
        42,
        "abcdef0123",
        55,
        model_init_seed=43,
        data_order_seed=44,
        train_runtime_seed=42,
    ) == "r6_dense_family_v1_k14g12_late_i43_d44_r42_step55_abcdef0"


def test_dense_run_name_supports_temporal_full112_prefix() -> None:
    assert run_name(
        "k1g1",
        42,
        "abcdef0123",
        55,
        model_init_seed=42,
        data_order_seed=42,
        train_runtime_seed=42,
        run_prefix="r7_dense_full112_temporal_v1",
    ) == "r7_dense_full112_temporal_v1_k1g1_i42_d42_r42_step55_abcdef0"


def test_load_probe_prompts_accepts_full112_and_deduplicates_prompts(tmp_path) -> None:
    probe = tmp_path / "probe.json"
    records = [{"text": f"probe-{index}"} for index in range(112)]
    probe.write_text(json.dumps({"records": records}), encoding="utf-8")
    payload, prompts = load_probe_prompts(probe)
    assert payload["records"] == records
    assert len(prompts) == 112

    duplicate_records = [{"text": "same"}, {"text": "same"}]
    probe.write_text(json.dumps({"records": duplicate_records}), encoding="utf-8")
    payload, prompts = load_probe_prompts(probe)
    assert payload["records"] == duplicate_records
    assert prompts == ["same"]


def test_copy_resume_artifacts_requires_complete_step55_run(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "checkpoint_step_000055.pt").write_bytes(b"checkpoint")
    (source / "training_metrics.csv").write_text("step\n55\n", encoding="utf-8")
    (source / "validation_metrics.csv").write_text("step\n", encoding="utf-8")
    chunks = source / "validation_chunks"
    chunks.mkdir()
    for step in dense_steps(55):
        (chunks / f"chunks_step_{step:06d}.json").write_text("{}", encoding="utf-8")

    destination = tmp_path / "destination"
    checkpoint = copy_resume_artifacts(source, destination)

    assert checkpoint == destination / "checkpoint_step_000055.pt"
    assert checkpoint.read_bytes() == b"checkpoint"
    assert len(list((destination / "validation_chunks").glob("*.json"))) == 5


def test_resume_seed_factors_support_legacy_and_split_metadata(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = source / "dense_run_config.json"
    config.write_text('{"seed": 42}', encoding="utf-8")
    assert resume_seed_factors(source) == {
        "model_init_seed": 42,
        "data_order_seed": 42,
        "train_runtime_seed": 42,
    }
    config.write_text(
        '{"seed": 42, "seed_factors": {'
        '"model_init_seed": 43, "data_order_seed": 44, '
        '"train_runtime_seed": 45}}',
        encoding="utf-8",
    )
    assert resume_seed_factors(source) == {
        "model_init_seed": 43,
        "data_order_seed": 44,
        "train_runtime_seed": 45,
    }


def test_resume_boundary_feature_supports_legacy_and_explicit_metadata(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = source / "dense_run_config.json"
    config.write_text('{"seed": 42}', encoding="utf-8")
    assert resume_boundary_feature(source) == {
        "mode": "final",
        "final_logit_bias": 2.0,
    }
    config.write_text(
        '{"boundary_feature": {'
        '"mode": "layer-scalar-mix", "final_logit_bias": 1.5}}',
        encoding="utf-8",
    )
    assert resume_boundary_feature(source) == {
        "mode": "layer-scalar-mix",
        "final_logit_bias": 1.5,
    }
