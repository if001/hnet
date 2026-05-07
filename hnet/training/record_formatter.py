import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

TextValue = str | int | float | bool | None

PREFERRED_TEXT_KEYS = (
    "text",
    "content",
    "completion",
    "response",
    "output",
    "answer",
    "prompt",
    "instruction",
    "question",
    "input",
    "context",
    "title",
)


class RecordFormatter(Protocol):
    def format_record(self, record: Mapping[str, object]) -> str | None: ...


def _stringify_scalar(value: TextValue) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _stringify_value(value: object) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _stringify_scalar(value)
    if isinstance(value, Mapping):
        return _stringify_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _stringify_sequence(value)
    return ""


def _join_message_content(messages: Sequence[object]) -> str:
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = _stringify_scalar(message.get("role"))
        content = _stringify_value(message.get("content"))
        text = content.strip()
        if not text:
            continue
        if role:
            parts.append(f"{role}: {text}")
        else:
            parts.append(text)
    return "\n".join(parts)


def _stringify_mapping(value: Mapping[str, object]) -> str:
    messages = value.get("messages")
    if isinstance(messages, Sequence) and not isinstance(
        messages, (str, bytes, bytearray)
    ):
        joined = _join_message_content(messages)
        if joined:
            return joined
    if "content" in value:
        return _stringify_value(value["content"])
    if "text" in value:
        return _stringify_value(value["text"])
    try:
        return json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
    except TypeError:
        return ""


def _stringify_sequence(value: Sequence[object]) -> str:
    if not value:
        return ""
    if all(isinstance(item, Mapping) for item in value):
        return _join_message_content(value)
    parts = [_stringify_value(item).strip() for item in value]
    return "\n".join(part for part in parts if part)


@dataclass(frozen=True)
class DefaultRecordFormatter:
    preferred_keys: tuple[str, ...] = PREFERRED_TEXT_KEYS

    def format_record(self, record: Mapping[str, object]) -> str | None:
        ordered_values: list[str] = []

        for key in self.preferred_keys:
            if key not in record:
                continue
            value = _stringify_value(record[key]).strip()
            if value:
                ordered_values.append(value)

        if not ordered_values:
            for key, value in record.items():
                if key.startswith("_"):
                    continue
                rendered = _stringify_value(value).strip()
                if rendered:
                    ordered_values.append(rendered)

        if not ordered_values:
            return None

        deduped_values = list(dict.fromkeys(ordered_values))
        return "\n\n".join(deduped_values)
