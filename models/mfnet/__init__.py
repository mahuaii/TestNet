from __future__ import annotations

from typing import Any

_MODEL_EXPORTS = {
    "UNetFormer": (".UNetFormer_MMSAM", "UNetFormer"),
    "UNetFormerDGA10": (".UNetFormer_MMSAM_dga10", "UNetFormerDGA10"),
    "UNetFormerDGA20": (".UNetFormer_MMSAM_dga20", "UNetFormerDGA20"),
    "UNetFormerDGA30": (".UNetFormer_MMSAM_dga30", "UNetFormerDGA30"),
    "UNetFormerDGA10ContributionStats": (
        ".UNetFormer_MMSAM_dga_contrib_stats",
        "UNetFormerDGA10ContributionStats",
    ),
    "UNetFormerDGA10ContributionStatsSoftplus": (
        ".UNetFormer_MMSAM_dga_contrib_stats",
        "UNetFormerDGA10ContributionStatsSoftplus",
    ),
    "UNetFormerDGA20ContributionStats": (
        ".UNetFormer_MMSAM_dga_contrib_stats",
        "UNetFormerDGA20ContributionStats",
    ),
    "UNetFormerDGA20ContributionStatsSoftplus": (
        ".UNetFormer_MMSAM_dga_contrib_stats",
        "UNetFormerDGA20ContributionStatsSoftplus",
    ),
    "UNetFormerPreAlign": (".UNetFormer_MMSAM_prealign", "UNetFormerPreAlign"),
    "UNetFormerPreAlignAuxAlign": (".UNetFormer_MMSAM_prealign_auxalign", "UNetFormerPreAlignAuxAlign"),
    "UNetFormerPreAlignDGA10": (".UNetFormer_MMSAM_prealign_dga10", "UNetFormerPreAlignDGA10"),
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
    "UNetFormerDGA10",
    "UNetFormerDGA20",
    "UNetFormerDGA30",
    "UNetFormerDGA10ContributionStats",
    "UNetFormerDGA10ContributionStatsSoftplus",
    "UNetFormerDGA20ContributionStats",
    "UNetFormerDGA20ContributionStatsSoftplus",
    "UNetFormerPreAlign",
    "UNetFormerPreAlignAuxAlign",
    "UNetFormerPreAlignDGA10",
]
