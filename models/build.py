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
    if model_type == "mfnet_unetformer_dga10":
        from .mfnet import UNetFormerDGA10

        model = UNetFormerDGA10(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "mfnet_unetformer_dga20":
        from .mfnet import UNetFormerDGA20

        model = UNetFormerDGA20(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "mfnet_unetformer_dga10_contrib_stats":
        from .mfnet import UNetFormerDGA10ContributionStats

        model = UNetFormerDGA10ContributionStats(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "mfnet_unetformer_dga20_contrib_stats":
        from .mfnet import UNetFormerDGA20ContributionStats

        model = UNetFormerDGA20ContributionStats(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "mfnet_unetformer_dga30":
        from .mfnet import UNetFormerDGA30

        model = UNetFormerDGA30(
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
    if model_type == "mfnet_unetformer_prealign_dga10":
        from .mfnet import UNetFormerPreAlignDGA10

        model = UNetFormerPreAlignDGA10(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    raise KeyError(
        "Unsupported model type: "
        f"{model_type!r}. Supported types: 'mfnet_unetformer', "
        "'mfnet_unetformer_dga10', 'mfnet_unetformer_dga20', "
        "'mfnet_unetformer_dga10_contrib_stats', 'mfnet_unetformer_dga20_contrib_stats', "
        "'mfnet_unetformer_dga30', "
        "'mfnet_unetformer_prealign', 'mfnet_unetformer_prealign_auxalign', "
        "'mfnet_unetformer_prealign_dga10'."
    )
