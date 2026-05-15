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


def _deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    for key, child_value in child.items():
        if key == "extends":
            continue

        parent_value = merged.get(key)
        if isinstance(parent_value, dict) and isinstance(child_value, dict):
            merged[key] = _deep_merge(parent_value, child_value)
        else:
            merged[key] = child_value
    return merged


def _load_config(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    if config_path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, config_path))
        raise ValueError(f"Config inheritance cycle detected: {cycle}")

    cfg = _loads_jsonc(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")

    parent_path = cfg.get("extends")
    if parent_path is None:
        return _deep_merge({}, cfg)
    if not isinstance(parent_path, str):
        raise ValueError(f"Config extends must be a string path: {config_path}")

    parent_config_path = Path(parent_path).expanduser()
    if not parent_config_path.is_absolute():
        parent_config_path = config_path.parent / parent_config_path

    parent_cfg = _load_config(parent_config_path, (*stack, config_path))
    return _deep_merge(parent_cfg, cfg)


def load_config(path: str) -> dict[str, Any]:
    return _load_config(Path(path), ())
