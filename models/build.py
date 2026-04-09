from __future__ import annotations

from typing import Any

def build_model(cfg: dict[str, Any]) -> Any:
    model_type = cfg["type"]
    if model_type == "mfnet_unetformer":
        from .mfnet import UNetFormer

        return UNetFormer(num_classes=int(cfg["num_classes"]))
    raise KeyError(
        "Unsupported model type: "
        f"{model_type!r}. Supported types: 'mfnet_unetformer'."
    )
