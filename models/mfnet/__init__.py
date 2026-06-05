from __future__ import annotations

from typing import Any

_MODEL_EXPORTS = {
    "UNetFormer": (".UNetFormer_MMSAM", "UNetFormer"),
    "UNetFormerMMAdapter10": (".UNetFormer_MMSAM", "UNetFormerMMAdapter10"),
    "UNetFormerMMAdapter20": (".UNetFormer_MMSAM", "UNetFormerMMAdapter20"),
    "UNetFormerMMAdapter21": (".UNetFormer_MMSAM", "UNetFormerMMAdapter21"),
    "UNetFormerDGA10": (".UNetFormer_MMSAM_dga10", "UNetFormerDGA10"),
    "UNetFormerDGA20": (".UNetFormer_MMSAM_dga20", "UNetFormerDGA20"),
    "UNetFormerDGA20DGSF10": (".UNetFormer_MMSAM_dga20_dgsf10", "UNetFormerDGA20DGSF10"),
    "UNetFormerDGSF10": (".UNetFormer_MMSAM_dgsf10", "UNetFormerDGSF10"),
    "UNetFormerDGFM": (".UNetFormer_MMSAM_dgfm", "UNetFormerDGFM"),
    "UNetFormerDGFM01": (".UNetFormer_MMSAM_dgfm01", "UNetFormerDGFM01"),
    "UNetFormerDGFM01UperNet": (".UNetFormer_MMSAM_dgfm01_upernet", "UNetFormerDGFM01UperNet"),
    "UNetFormerSGCF": (".UNetFormer_MMSAM_sgcf", "UNetFormerSGCF"),
    "UNetFormerDGA30": (".UNetFormer_MMSAM_dga30", "UNetFormerDGA30"),
    "UNetFormerDGA10Softplus": (".UNetFormer_MMSAM_dga_softplus", "UNetFormerDGA10Softplus"),
    "UNetFormerDGA20Softplus": (".UNetFormer_MMSAM_dga_softplus", "UNetFormerDGA20Softplus"),
    "UNetFormerAuxAlign": (".UNetFormer_MMSAM_auxalign", "UNetFormerAuxAlign"),
    "UNetFormerPreAlign": (".UNetFormer_MMSAM_prealign", "UNetFormerPreAlign"),
    "UNetFormerPreAlignMMAdapter10": (
        ".UNetFormer_MMSAM_prealign_mmadapter10",
        "UNetFormerPreAlignMMAdapter10",
    ),
    "UNetFormerPreAlignMMAdapter20": (
        ".UNetFormer_MMSAM_prealign_mmadapter20",
        "UNetFormerPreAlignMMAdapter20",
    ),
    "UNetFormerPreAlignMMAdapter21": (
        ".UNetFormer_MMSAM_prealign_mmadapter21",
        "UNetFormerPreAlignMMAdapter21",
    ),
    "UNetFormerPreAlignAuxAlign": (".UNetFormer_MMSAM_prealign_auxalign", "UNetFormerPreAlignAuxAlign"),
    "UNetFormerPreAlignDGA10": (".UNetFormer_MMSAM_prealign_dga10", "UNetFormerPreAlignDGA10"),
    "UNetFormerPreAlignAuxAlignDGA10": (
        ".UNetFormer_MMSAM_prealign_auxalign_dga10",
        "UNetFormerPreAlignAuxAlignDGA10",
    ),
    "UNetFormerPreAlignAuxAlignDGSF10": (
        ".UNetFormer_MMSAM_prealign_auxalign_dgsf10",
        "UNetFormerPreAlignAuxAlignDGSF10",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _MODEL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _MODEL_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "UNetFormer",
    "UNetFormerMMAdapter10",
    "UNetFormerMMAdapter20",
    "UNetFormerMMAdapter21",
    "UNetFormerDGA10",
    "UNetFormerDGA20",
    "UNetFormerDGA20DGSF10",
    "UNetFormerDGSF10",
    "UNetFormerDGFM",
    "UNetFormerDGFM01",
    "UNetFormerDGFM01UperNet",
    "UNetFormerSGCF",
    "UNetFormerDGA30",
    "UNetFormerDGA10Softplus",
    "UNetFormerDGA20Softplus",
    "UNetFormerAuxAlign",
    "UNetFormerPreAlign",
    "UNetFormerPreAlignMMAdapter10",
    "UNetFormerPreAlignMMAdapter20",
    "UNetFormerPreAlignMMAdapter21",
    "UNetFormerPreAlignAuxAlign",
    "UNetFormerPreAlignDGA10",
    "UNetFormerPreAlignAuxAlignDGA10",
    "UNetFormerPreAlignAuxAlignDGSF10",
]
