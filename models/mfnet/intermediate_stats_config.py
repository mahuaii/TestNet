from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from utils import IntermediateStatsRecorder


def ensure_intermediate_stats_recorder(owner: Any) -> IntermediateStatsRecorder:
    stats = getattr(owner, "intermediate_stats", None)
    if stats is None:
        stats = IntermediateStatsRecorder()
        owner.intermediate_stats = stats
    return stats


def attach_intermediate_stats(owner: Any, module: Any, prefix: str) -> None:
    module.intermediate_stats = ensure_intermediate_stats_recorder(owner)
    module.intermediate_stats_prefix = prefix


def attach_requested_intermediate_stats(
    owner: Any,
    requested_modules: Iterable[str],
    available_modules: Mapping[str, Iterable[tuple[Any, str]]],
) -> None:
    for module_name in requested_modules:
        for module, prefix in available_modules.get(module_name.lower(), ()):
            attach_intermediate_stats(owner, module, prefix)


__all__ = [
    "attach_intermediate_stats",
    "attach_requested_intermediate_stats",
    "ensure_intermediate_stats_recorder",
]
