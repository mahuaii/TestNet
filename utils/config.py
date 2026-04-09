from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _strip_jsonc_comments(text: str) -> str:
    result: list[str] = []
    in_string = False
    string_delimiter = ""
    escape = False
    in_line_comment = False
    in_block_comment = False
    i = 0

    while i < len(text):
        char = text[i]
        next_char = text[i + 1] if i + 1 < len(text) else ""

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
                result.append(char)
            i += 1
            continue

        if in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == string_delimiter:
                in_string = False
            i += 1
            continue

        if char in {'"', "'"}:
            in_string = True
            string_delimiter = char
            result.append(char)
            i += 1
            continue

        if char == "/" and next_char == "/":
            in_line_comment = True
            i += 2
            continue

        if char == "/" and next_char == "*":
            in_block_comment = True
            i += 2
            continue

        result.append(char)
        i += 1

    return "".join(result)


def _remove_trailing_commas(text: str) -> str:
    result: list[str] = []
    in_string = False
    string_delimiter = ""
    escape = False
    i = 0

    while i < len(text):
        char = text[i]

        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == string_delimiter:
                in_string = False
            i += 1
            continue

        if char in {'"', "'"}:
            in_string = True
            string_delimiter = char
            result.append(char)
            i += 1
            continue

        if char == ",":
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] in "}]":
                i += 1
                continue

        result.append(char)
        i += 1

    return "".join(result)


def _loads_jsonc(text: str) -> Any:
    return json.loads(_remove_trailing_commas(_strip_jsonc_comments(text)))


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    cfg = _loads_jsonc(config_path.read_text(encoding="utf-8"))

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return cfg
