from __future__ import annotations

import torch
import torch.nn as nn

from .UNetFormer_MMSAM_spmf20 import UNetFormerSPMF20
from .modules import AuxPreAlign


class UNetFormerPreAlignSPMF20(UNetFormerSPMF20):
    def __init__(self, *args: object, aux_in_channels: int = 1, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.aux_prealign = AuxPreAlign(in_channels=aux_in_channels, out_channels=3)
        nn.init.zeros_(self.aux_prealign.project.weight)
        nn.init.zeros_(self.aux_prealign.project.bias)

    def _make_sam_dsm_input(self, dsm: torch.Tensor) -> torch.Tensor:
        return self.aux_prealign(dsm)


__all__ = ["UNetFormerPreAlignSPMF20"]
