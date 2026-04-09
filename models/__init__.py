from .build import build_model

__all__ = ["build_model", "UNetFormer"]


def __getattr__(name: str):
    if name == "UNetFormer":
        from .mfnet import UNetFormer

        return UNetFormer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
