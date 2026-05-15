from __future__ import annotations

import torch.nn as nn

from utils import IntermediateStatsRecorder

from .UNetFormer_MMSAM_dga10 import UNetFormerDGA10
from .UNetFormer_MMSAM_dga20 import UNetFormerDGA20
from .modules import DGABlock10Softplus, DGABlock20Softplus


class UNetFormerDGA10Softplus(UNetFormerDGA10):
    def __init__(self, *args: object, record_intermediate_stats: bool = False, **kwargs: object) -> None:
        super().__init__(*args, record_intermediate_stats=False, **kwargs)
        self.dga_blocks = nn.ModuleList(
            [DGABlock10Softplus(channels=int(self.image_encoder.embed_dim)) for _ in self.dga_indexes]
        )
        if record_intermediate_stats:
            self.intermediate_stats = IntermediateStatsRecorder()
            self._attach_intermediate_stats_to_dga_blocks()


class UNetFormerDGA20Softplus(UNetFormerDGA20):
    def __init__(self, *args: object, record_intermediate_stats: bool = False, **kwargs: object) -> None:
        super().__init__(*args, record_intermediate_stats=False, **kwargs)
        self.dga_blocks = nn.ModuleList(
            [DGABlock20Softplus(channels=int(self.image_encoder.embed_dim)) for _ in self.dga_indexes]
        )
        if record_intermediate_stats:
            self.intermediate_stats = IntermediateStatsRecorder()
            self._attach_intermediate_stats_to_dga_blocks()


__all__ = ["UNetFormerDGA10Softplus", "UNetFormerDGA20Softplus"]
