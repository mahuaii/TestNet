from __future__ import annotations

from typing import Any

from .segmentor import RGBSegmentor


def build_model(cfg: dict[str, Any]) -> RGBSegmentor:
    model_cls = {
        "rgb_segmentor": RGBSegmentor,
    }[cfg["type"]]
    return model_cls()
