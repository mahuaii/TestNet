from __future__ import annotations

import torch
import torch.nn as nn

from .UNetFormer_MMSAM_spmf20 import UNetFormerSPMF20, _as_single_channel_dsm
from .modules import AuxPreAlign


class UNetFormerPreAlignSPMF20(UNetFormerSPMF20):
    def __init__(self, *args: object, aux_in_channels: int = 1, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.aux_prealign = AuxPreAlign(in_channels=aux_in_channels, out_channels=3)
        nn.init.zeros_(self.aux_prealign.project.weight)
        nn.init.zeros_(self.aux_prealign.project.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str = "Train") -> torch.Tensor:
        del mode
        h, w = x.size()[-2:]
        dsm = _as_single_channel_dsm(y)
        y_sam = self.aux_prealign(dsm)

        deepx, deepy, dsm_taps = self._encode_spmf20_features(x, y_sam)
        rgb_feats = (
            self.fpn1x(deepx),
            self.fpn2x(deepx),
            self.fpn3x(deepx),
            self.fpn4x(deepx),
        )
        dsm_feats = (
            self.fpn1y(deepy),
            self.fpn2y(deepy),
            self.fpn3y(deepy),
            self.fpn4y(deepy),
        )
        structure_feats = self.structure_branch10(dsm, dsm_taps)
        res1, res2, res3, res4 = self.spmf20(rgb_feats, dsm_feats, structure_feats)
        return self.decoder(res1, res2, res3, res4, h, w)


__all__ = ["UNetFormerPreAlignSPMF20"]
