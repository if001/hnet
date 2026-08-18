from scripts.evaluate_agent_proxy import (
    extract_first_json_object,
    render_prompt,
    score_response,
)


def test_render_prompt_matches_sft_tool_envelope():
    prompt = render_prompt(
        {
            "tools": [{"name": "calculator", "arguments": {}}],
            "user": "Compute 2 + 2.",
        }
    )
    assert "You are a helpful assistant.\n/no_think\n<tools>\n" in prompt
    assert "Available tools:" not in prompt
    assert "<|im_start|>user\nCompute 2 + 2.\n" in prompt


def test_extract_first_json_object_skips_prefix_and_handles_nested_arguments():
    text = 'answer: {"tool":"search_docs","arguments":{"query":"a}b","top_k":3}} trailing'
    assert extract_first_json_object(text) == {
        "tool": "search_docs",
        "arguments": {"query": "a}b", "top_k": 3},
    }


def test_score_response_requires_exact_tool_and_arguments():
    task = {
        "expected_tool": "calculator",
        "expected_arguments": {"expression": "2 + 2"},
    }
    score = score_response(
        task, '{"tool":"calculator","arguments":{"expression":"2 + 2"}}'
    )
    assert score["valid_json"]
    assert score["full_exact"]


def test_score_response_accepts_sft_name_schema_and_tool_call_envelope():
    task = {
        "expected_tool": "calculator",
        "expected_arguments": {"expression": "2 + 2"},
    }
    score = score_response(
        task,
        '<tool_call>\n{"name":"calculator","arguments":{"expression":"2 + 2"}}\n</tool_call>',
    )
    assert score["valid_json"]
    assert score["full_exact"]


def test_invalid_json_scores_false():
    score = score_response(
        {"expected_tool": "calculator", "expected_arguments": {}},
        "calculator(expression=2+2)",
    )
    assert not score["valid_json"]
    assert not score["full_exact"]
