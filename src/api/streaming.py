import json
from typing import Any


class AnswerStreamExtractor:
    """Extracts the value of the top-level JSON `answer` string from streamed chunks."""

    def __init__(self) -> None:
        self._state = "search_key"
        self._key_buffer = ""
        self._escape = False
        self._unicode_buffer: str | None = None

    @property
    def done(self) -> bool:
        return self._state == "done"

    def feed(self, chunk: str) -> str:
        output: list[str] = []

        for char in chunk:
            if self._state == "search_key":
                self._key_buffer = (self._key_buffer + char)[-8:]
                if self._key_buffer.endswith('"answer"'):
                    self._state = "await_colon"
                    self._key_buffer = ""
                continue

            if self._state == "await_colon":
                if char == ":":
                    self._state = "await_value"
                continue

            if self._state == "await_value":
                if char == '"':
                    self._state = "in_answer"
                continue

            if self._state != "in_answer":
                continue

            if self._unicode_buffer is not None:
                self._unicode_buffer += char
                if len(self._unicode_buffer) == 4:
                    try:
                        output.append(chr(int(self._unicode_buffer, 16)))
                    except ValueError:
                        output.append(f"\\u{self._unicode_buffer}")
                    self._unicode_buffer = None
                continue

            if self._escape:
                if char == "u":
                    self._unicode_buffer = ""
                else:
                    output.append(
                        {
                            '"': '"',
                            "\\": "\\",
                            "/": "/",
                            "b": "\b",
                            "f": "\f",
                            "n": "\n",
                            "r": "\r",
                            "t": "\t",
                        }.get(char, char)
                    )
                self._escape = False
                continue

            if char == "\\":
                self._escape = True
                continue

            if char == '"':
                self._state = "done"
                continue

            output.append(char)

        return "".join(output)


def normalize_stream_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def ndjson_event(event_type: str, **payload: Any) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


def normalize_suggestions(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()][:3]
