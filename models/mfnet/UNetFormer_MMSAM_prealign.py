from __future__ import annotations

import torch

from .UNetFormer_MMSAM import UNetFormer
from .modules import AuxPreAlign


class UNetFormerPreAlign(UNetFormer):
    def __init__(self, *args, aux_in_channels: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.aux_prealign = AuxPreAlign(in_channels=aux_in_channels, out_channels=3)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str = "Train") -> torch.Tensor:
        del mode
        h, w = x.size()[-2:]
        if y.ndim == 3:
            y = y.unsqueeze(1)
        y = self.aux_prealign(y)
        deepx, deepy = self.image_encoder(x, y)

        res1x = self.fpn1x(deepx)
        res2x = self.fpn2x(deepx)
        res3x = self.fpn3x(deepx)
        res4x = self.fpn4x(deepx)
        res1y = self.fpn1y(deepy)
        res2y = self.fpn2y(deepy)
        res3y = self.fpn3y(deepy)
        res4y = self.fpn4y(deepy)
        res1 = self.fusion1(res1x, res1y)
        res2 = self.fusion2(res2x, res2y)
        res3 = self.fusion3(res3x, res3y)
        res4 = self.fusion4(res4x, res4y)
        return self.decoder(res1, res2, res3, res4, h, w)
