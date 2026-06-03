from __future__ import annotations

from collections.abc import Iterable

import torch

from .UNetFormer_MMSAM_dgfm import _encode_dgfm_decoder_features, _init_dgfm_modules
from .UNetFormer_MMSAM_prealign import UNetFormerPreAlign


class UNetFormerPreAlignDGFM(UNetFormerPreAlign):
    def __init__(
        self,
        *args: object,
        record_intermediate_stats: bool = False,
        record_intermediate_modules: Iterable[str] = (),
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        _init_dgfm_modules(self, record_intermediate_stats, record_intermediate_modules)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str = "Train") -> torch.Tensor:
        del mode
        h, w = x.size()[-2:]
        if y.ndim == 3:
            y = y.unsqueeze(1)
        y = self.aux_prealign(y)
        res1, res2, res3, res4 = _encode_dgfm_decoder_features(self, x, y)
        return self.decoder(res1, res2, res3, res4, h, w)


__all__ = ["UNetFormerPreAlignDGFM"]
