from __future__ import annotations

import time


class AnchorTimer:
    def __init__(self) -> None:
        self._anchors: dict[str, float] = {}

    def mark(self, name: str) -> None:
        self._anchors[name] = time.perf_counter()

    def has(self, name: str) -> bool:
        return name in self._anchors

    def elapsed(self, name: str) -> float:
        return time.perf_counter() - self._anchors[name]
