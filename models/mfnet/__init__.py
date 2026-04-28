from __future__ import annotations

from typing import Any

_MODEL_EXPORTS = {
    "UNetFormer": (".UNetFormer_MMSAM", "UNetFormer"),
    "UNetFormerPreAlign": (".UNetFormer_MMSAM_prealign", "UNetFormerPreAlign"),
    "UNetFormerPreAlignAuxAlign": (".UNetFormer_MMSAM_prealign_auxalign", "UNetFormerPreAlignAuxAlign"),
    "UNetFormerPreAlignDGA": (".UNetFormer_MMSAM_prealign_dga", "UNetFormerPreAlignDGA"),
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
    "UNetFormerPreAlign",
    "UNetFormerPreAlignAuxAlign",
    "UNetFormerPreAlignDGA",
]
