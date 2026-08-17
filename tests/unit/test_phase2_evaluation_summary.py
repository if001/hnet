import csv

from scripts.summarize_phase2_evaluations import (
    aggregate_model_category,
    collect_agent,
    seed_from_name,
)


def test_seed_from_pretraining_and_sft_run_names():
    assert seed_from_name("t26_95m_ctx_s42_commit") == 42
    assert seed_from_name("t26_pre_s43_sft_s42_ctx8192_commit") == 43


def test_collect_and_aggregate_agent_summary(tmp_path):
    run = tmp_path / "t26" / "t26_pre_s42_sft_s42_ctx8192_commit"
    run.mkdir(parents=True)
    path = run / "agent_proxy_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["category", "tasks", "json_valid_rate", "tool_accuracy",
                        "argument_exact_rate", "full_exact_rate"],
        )
        writer.writeheader()
        writer.writerow(
            {"category": "__overall__", "tasks": 10, "json_valid_rate": 0.5,
             "tool_accuracy": 0.4, "argument_exact_rate": 0.3, "full_exact_rate": 0.2}
        )
    rows = collect_agent(tmp_path)
    summary = aggregate_model_category(
        rows, ["json_valid_rate", "full_exact_rate"], seed_key="pretraining_seed"
    )
    assert summary[0]["seed_count"] == 1
    assert summary[0]["full_exact_rate_mean"] == 0.2
