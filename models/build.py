from __future__ import annotations

from typing import Any


def _record_intermediate_stats(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("record_intermediate_stats", False))


def _intermediate_stats_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"record_intermediate_stats": _record_intermediate_stats(cfg)}
    if "record_intermediate_modules" in cfg:
        kwargs["record_intermediate_modules"] = cfg["record_intermediate_modules"]
    return kwargs


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
    if model_type == "mfnet_unetformer_mmadapter10":
        from .mfnet import UNetFormerMMAdapter10

        model = UNetFormerMMAdapter10(
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
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    if model_type == "mfnet_unetformer_dga20":
        from .mfnet import UNetFormerDGA20

        model = UNetFormerDGA20(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    if model_type == "mfnet_unetformer_dga20_dgsf10":
        from .mfnet import UNetFormerDGA20DGSF10

        model = UNetFormerDGA20DGSF10(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    if model_type == "mfnet_unetformer_dgsf10":
        from .mfnet import UNetFormerDGSF10

        model = UNetFormerDGSF10(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    if model_type == "mfnet_unetformer_dga10_softplus":
        from .mfnet import UNetFormerDGA10Softplus

        model = UNetFormerDGA10Softplus(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    if model_type == "mfnet_unetformer_dga20_softplus":
        from .mfnet import UNetFormerDGA20Softplus

        model = UNetFormerDGA20Softplus(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
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
    if model_type == "mfnet_unetformer_prealign_auxalign_dga10":
        from .mfnet import UNetFormerPreAlignAuxAlignDGA10

        model = UNetFormerPreAlignAuxAlignDGA10(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "mfnet_unetformer_prealign_auxalign_dgsf10":
        from .mfnet import UNetFormerPreAlignAuxAlignDGSF10

        model = UNetFormerPreAlignAuxAlignDGSF10(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    raise KeyError(
        "Unsupported model type: "
        f"{model_type!r}. Supported types: 'mfnet_unetformer', "
        "'mfnet_unetformer_mmadapter10', "
        "'mfnet_unetformer_dga10', 'mfnet_unetformer_dga20', "
        "'mfnet_unetformer_dga20_dgsf10', 'mfnet_unetformer_dgsf10', "
        "'mfnet_unetformer_dga10_softplus', 'mfnet_unetformer_dga20_softplus', "
        "'mfnet_unetformer_dga30', "
        "'mfnet_unetformer_prealign', 'mfnet_unetformer_prealign_auxalign', "
        "'mfnet_unetformer_prealign_dga10', "
        "'mfnet_unetformer_prealign_auxalign_dga10', "
        "'mfnet_unetformer_prealign_auxalign_dgsf10'."
    )
