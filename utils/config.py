from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def resolve_config_path(path: str | Path) -> Path:
    """Resolve a config path or an exact config filename directly under ``configs``."""
    requested_path = Path(path).expanduser()
    if requested_path.is_file():
        return requested_path.resolve()

    if requested_path.is_absolute():
        raise FileNotFoundError(f"Config file not found: {requested_path}")

    config_path = CONFIGS_DIR / requested_path.name
    if config_path.is_file():
        return config_path.resolve()

    raise FileNotFoundError(
        f"Config file {path!r} not found. Searched directly under {CONFIGS_DIR}."
    )


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


def _resolve_parent_paths(extends: Any, config_path: Path) -> list[Path]:
    if extends is None:
        return []

    if isinstance(extends, str):
        parent_paths = [extends]
    elif isinstance(extends, list):
        if not all(isinstance(parent_path, str) for parent_path in extends):
            raise ValueError(
                f"Config extends entries must be string paths: {config_path}"
            )
        parent_paths = extends
    else:
        raise ValueError(
            f"Config extends must be a string path or list of string paths: {config_path}"
        )

    resolved_paths: list[Path] = []
    for parent_path in parent_paths:
        resolved_parent_path = Path(parent_path).expanduser()
        if not resolved_parent_path.is_absolute():
            resolved_parent_path = config_path.parent / resolved_parent_path
        resolved_paths.append(resolved_parent_path)
    return resolved_paths


def _load_config(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    if config_path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, config_path))
        raise ValueError(f"Config inheritance cycle detected: {cycle}")

    cfg = _loads_jsonc(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")

    parent_paths = _resolve_parent_paths(cfg.get("extends"), config_path)
    if not parent_paths:
        return _deep_merge({}, cfg)

    merged_parents: dict[str, Any] = {}
    # Merge parents in declaration order so later parents have higher priority.
    for parent_path in parent_paths:
        parent_cfg = _load_config(parent_path, (*stack, config_path))
        merged_parents = _deep_merge(merged_parents, parent_cfg)
    return _deep_merge(merged_parents, cfg)


def load_config(path: str | Path) -> dict[str, Any]:
    return _load_config(resolve_config_path(path), ())
