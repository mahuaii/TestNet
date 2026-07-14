from __future__ import annotations

from typing import Any

AVAILABLE_MODEL_TYPES: tuple[str, ...] = (
    "mfnet_unetformer",
    "testnet_mmadapter10",
    "testnet_mmadapter20",
    "testnet_mmadapter21",
    "testnet_dga10",
    "testnet_dga20",
    "testnet_dga20_dgsf10",
    "testnet_dgsf10",
    "testnet_dgfm",
    "testnet_dgfm01",
    "testnet_dgfm01_upernet",
    "testnet_sgcf",
    "testnet_spmf10",
    "testnet_spmf11",
    "testnet_spmf20",
    "testnet_spmf21",
    "testnet_spmf22",
    "testnet_prealign_spmf20",
    "testnet_prealign_spmf21",
    "testnet_dga10_softplus",
    "testnet_dga20_softplus",
    "testnet_dga30",
    "testnet_auxalign",
    "testnet_prealign",
    "testnet_prealign_mmadapter10",
    "testnet_prealign_mmadapter20",
    "testnet_prealign_mmadapter21",
    "testnet_prealign_auxalign",
    "testnet_prealign_dga10",
    "testnet_prealign_auxalign_dga10",
    "testnet_prealign_auxalign_dgsf10",
)


def _record_intermediate_stats(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("record_intermediate_stats", False))


def _intermediate_stats_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"record_intermediate_stats": _record_intermediate_stats(cfg)}
    if "record_intermediate_modules" in cfg:
        kwargs["record_intermediate_modules"] = cfg["record_intermediate_modules"]
    return kwargs


def _spmf_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    detach_dsm_taps = cfg.get("detach_dsm_taps", True)
    if not isinstance(detach_dsm_taps, bool):
        raise TypeError(f"model.detach_dsm_taps must be a bool, got {type(detach_dsm_taps).__name__}.")
    return {**_intermediate_stats_kwargs(cfg), "detach_dsm_taps": detach_dsm_taps}


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
    if model_type == "testnet_mmadapter10":
        from .mfnet import UNetFormerMMAdapter10

        model = UNetFormerMMAdapter10(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "testnet_mmadapter20":
        from .mfnet import UNetFormerMMAdapter20

        model = UNetFormerMMAdapter20(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "testnet_mmadapter21":
        from .mfnet import UNetFormerMMAdapter21

        model = UNetFormerMMAdapter21(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "testnet_dga10":
        from .mfnet import UNetFormerDGA10

        model = UNetFormerDGA10(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    if model_type == "testnet_dga20":
        from .mfnet import UNetFormerDGA20

        model = UNetFormerDGA20(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    if model_type == "testnet_dga20_dgsf10":
        from .mfnet import UNetFormerDGA20DGSF10

        model = UNetFormerDGA20DGSF10(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    if model_type == "testnet_dgsf10":
        from .mfnet import UNetFormerDGSF10

        model = UNetFormerDGSF10(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    if model_type == "testnet_dgfm":
        from .mfnet import UNetFormerDGFM

        model = UNetFormerDGFM(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    if model_type == "testnet_dgfm01":
        from .mfnet import UNetFormerDGFM01

        model = UNetFormerDGFM01(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    if model_type == "testnet_dgfm01_upernet":
        from .mfnet import UNetFormerDGFM01UperNet

        model = UNetFormerDGFM01UperNet(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    if model_type == "testnet_sgcf":
        from .mfnet import UNetFormerSGCF

        model = UNetFormerSGCF(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    if model_type == "testnet_spmf10":
        from .mfnet import UNetFormerSPMF10

        model = UNetFormerSPMF10(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_spmf_kwargs(cfg),
        )
        return model
    if model_type == "testnet_spmf11":
        from .mfnet import UNetFormerSPMF11

        model = UNetFormerSPMF11(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_spmf_kwargs(cfg),
        )
        return model
    if model_type == "testnet_spmf20":
        from .mfnet import UNetFormerSPMF20

        model = UNetFormerSPMF20(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_spmf_kwargs(cfg),
        )
        return model
    if model_type == "testnet_spmf21":
        from .mfnet import UNetFormerSPMF21

        model = UNetFormerSPMF21(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_spmf_kwargs(cfg),
        )
        return model
    if model_type == "testnet_spmf22":
        from .mfnet import UNetFormerSPMF22

        model = UNetFormerSPMF22(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_spmf_kwargs(cfg),
        )
        return model
    if model_type == "testnet_prealign_spmf20":
        from .mfnet import UNetFormerPreAlignSPMF20

        model = UNetFormerPreAlignSPMF20(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_spmf_kwargs(cfg),
        )
        return model
    if model_type == "testnet_prealign_spmf21":
        from .mfnet import UNetFormerPreAlignSPMF21

        model = UNetFormerPreAlignSPMF21(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_spmf_kwargs(cfg),
        )
        return model
    if model_type == "testnet_dga10_softplus":
        from .mfnet import UNetFormerDGA10Softplus

        model = UNetFormerDGA10Softplus(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    if model_type == "testnet_dga20_softplus":
        from .mfnet import UNetFormerDGA20Softplus

        model = UNetFormerDGA20Softplus(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    if model_type == "testnet_dga30":
        from .mfnet import UNetFormerDGA30

        model = UNetFormerDGA30(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "testnet_auxalign":
        from .mfnet import UNetFormerAuxAlign

        model = UNetFormerAuxAlign(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "testnet_prealign":
        from .mfnet import UNetFormerPreAlign

        model = UNetFormerPreAlign(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "testnet_prealign_mmadapter10":
        from .mfnet import UNetFormerPreAlignMMAdapter10

        model = UNetFormerPreAlignMMAdapter10(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "testnet_prealign_mmadapter20":
        from .mfnet import UNetFormerPreAlignMMAdapter20

        model = UNetFormerPreAlignMMAdapter20(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "testnet_prealign_mmadapter21":
        from .mfnet import UNetFormerPreAlignMMAdapter21

        model = UNetFormerPreAlignMMAdapter21(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "testnet_prealign_auxalign":
        from .mfnet import UNetFormerPreAlignAuxAlign

        model = UNetFormerPreAlignAuxAlign(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "testnet_prealign_dga10":
        from .mfnet import UNetFormerPreAlignDGA10

        model = UNetFormerPreAlignDGA10(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "testnet_prealign_auxalign_dga10":
        from .mfnet import UNetFormerPreAlignAuxAlignDGA10

        model = UNetFormerPreAlignAuxAlignDGA10(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
        )
        return model
    if model_type == "testnet_prealign_auxalign_dgsf10":
        from .mfnet import UNetFormerPreAlignAuxAlignDGSF10

        model = UNetFormerPreAlignAuxAlignDGSF10(
            num_classes=int(cfg["num_classes"]),
            sam_backbone=str(cfg["sam_backbone"]),
            sam_checkpoint=str(cfg["sam_checkpoint"]),
            **_intermediate_stats_kwargs(cfg),
        )
        return model
    supported_types = ", ".join(repr(model_type) for model_type in AVAILABLE_MODEL_TYPES)
    raise KeyError(f"Unsupported model type: {model_type!r}. Supported types: {supported_types}.")
