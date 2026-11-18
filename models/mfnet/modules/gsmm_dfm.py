from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from ..UNetFormer_MMSAM import SEFusion
from ._spmf_common import as_four_tuple


class MultiScaleGSMMDFMFusion(nn.Module):
    """Add GSMM features to both modalities before the original per-scale DFM."""

    def __init__(self, channels: int | Sequence[int] = 256) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            SEFusion(channel) for channel in as_four_tuple("channels", channels)
        )

    def forward(
        self,
        rgb_feats: Sequence[torch.Tensor],
        dsm_feats: Sequence[torch.Tensor],
        structure_feats: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb1, rgb2, rgb3, rgb4 = rgb_feats
        dsm1, dsm2, dsm3, dsm4 = dsm_feats
        s1, s2, s3, s4 = structure_feats
        return (
            self.blocks[0](rgb1 + s1, dsm1 + s1),
            self.blocks[1](rgb2 + s2, dsm2 + s2),
            self.blocks[2](rgb3 + s3, dsm3 + s3),
            self.blocks[3](rgb4 + s4, dsm4 + s4),
        )


__all__ = ["MultiScaleGSMMDFMFusion"]
