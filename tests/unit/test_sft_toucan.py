import json

from hnet.sft.dataset import _map_toucan


def test_map_toucan_preserves_tools_and_merges_parallel_calls() -> None:
    example = {
        "tools": json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        ),
        "messages": json.dumps(
            [
                {"role": "user", "content": "調べて"},
                {"role": "assistant", "content": "確認します。"},
                {
                    "role": "tool_call",
                    "content": "{'name': 'search', 'arguments': '{\"q\": \"A\"}'}",
                },
                {
                    "role": "tool_call",
                    "content": "{'name': 'search', 'arguments': '{\"q\": \"B\"}'}",
                },
                {"role": "tool_response", "content": "result A"},
                {"role": "tool_response", "content": "result B"},
                {"role": "assistant", "content": "完了しました。"},
            ],
            ensure_ascii=False,
        ),
    }

    mapped = _map_toucan(example, "You are helpful.")

    assert [message["role"] for message in mapped["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    system = mapped["messages"][0]["content"]
    assert "/no_think" in system
    assert '"name":"search"' in system
    calls = mapped["messages"][2]["content"]
    assert calls.count("<tool_call>") == 2
    assert '"arguments":{"q":"A"}' in calls
    responses = mapped["messages"][3]["content"]
    assert responses.count("<tool_response>") == 2


def test_map_toucan_rejects_unknown_roles() -> None:
    example = {
        "tools": "[]",
        "messages": json.dumps(
            [
                {"role": "user", "content": "hello"},
                {"role": "unknown", "content": "bad"},
            ]
        ),
    }

    try:
        _map_toucan(example, "")
    except ValueError as exc:
        assert "Unsupported Toucan role" in str(exc)
    else:
        raise AssertionError("unknown Toucan roles must be rejected")
