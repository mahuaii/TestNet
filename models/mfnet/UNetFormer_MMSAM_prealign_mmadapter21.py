from __future__ import annotations

from .UNetFormer_MMSAM_prealign import UNetFormerPreAlign


class UNetFormerPreAlignMMAdapter21(UNetFormerPreAlign):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, mm_adapter_block="mmadapter21", **kwargs)


__all__ = ["UNetFormerPreAlignMMAdapter21"]
