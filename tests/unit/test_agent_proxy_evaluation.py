from scripts.evaluate_agent_proxy import extract_first_json_object, score_response


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


def test_invalid_json_scores_false():
    score = score_response(
        {"expected_tool": "calculator", "expected_arguments": {}},
        "calculator(expression=2+2)",
    )
    assert not score["valid_json"]
    assert not score["full_exact"]
