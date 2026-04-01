from hnet.training.data import DefaultRecordFormatter


def test_default_record_formatter_prefers_known_text_fields() -> None:
    formatter = DefaultRecordFormatter()

    record = {
        "instruction": "Explain Japanese grammar.",
        "output": "This is the answer.",
        "ignored": 123,
    }

    assert formatter.format_record(record) == "This is the answer.\n\nExplain Japanese grammar."


def test_default_record_formatter_handles_message_lists() -> None:
    formatter = DefaultRecordFormatter()

    record = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    }

    assert formatter.format_record(record) == "user: Hi\nassistant: Hello"
