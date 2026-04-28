from __future__ import annotations

from typing import Any


def build_model(cfg: dict[str, Any]) -> Any:
    model_type = cfg["type"]
    if model_type == "mfnet_unetformer":
        from .mfnet import UNetFormer

        model = UNetFormer(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "mfnet_unetformer_dga":
        from .mfnet import UNetFormerDGA

        model = UNetFormerDGA(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "mfnet_unetformer_prealign":
        from .mfnet import UNetFormerPreAlign

        model = UNetFormerPreAlign(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "mfnet_unetformer_prealign_auxalign":
        from .mfnet import UNetFormerPreAlignAuxAlign

        model = UNetFormerPreAlignAuxAlign(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "mfnet_unetformer_prealign_dga":
        from .mfnet import UNetFormerPreAlignDGA

        model = UNetFormerPreAlignDGA(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    raise KeyError(
        "Unsupported model type: "
        f"{model_type!r}. Supported types: 'mfnet_unetformer', "
        "'mfnet_unetformer_dga', "
        "'mfnet_unetformer_prealign', 'mfnet_unetformer_prealign_auxalign', "
        "'mfnet_unetformer_prealign_dga'."
    )
