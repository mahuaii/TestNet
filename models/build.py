from __future__ import annotations

from typing import Any


def build_model(cfg: dict[str, Any]) -> Any:
    model_type = cfg["type"]
    if model_type == "mfnet_unetformer":
        from .mfnet import UNetFormer

        model = UNetFormer(
            num_classes=int(cfg["num_classes"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    raise KeyError(
        "Unsupported model type: "
        f"{model_type!r}. Supported types: 'mfnet_unetformer'."
    )
