from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn

from .UNetFormer_MMSAM_dga10 import UNetFormerDGA10
from .UNetFormer_MMSAM_dga20 import UNetFormerDGA20
from .intermediate_stats_config import attach_requested_intermediate_stats
from .modules import DGABlock10Softplus, DGABlock20Softplus


class UNetFormerDGA10Softplus(UNetFormerDGA10):
    def __init__(
        self,
        *args: object,
        record_intermediate_stats: bool = False,
        record_intermediate_modules: Iterable[str] = (),
        **kwargs: object,
    ) -> None:
        super().__init__(
            *args,
            record_intermediate_stats=False,
            record_intermediate_modules=(),
            **kwargs,
        )
        self.dga_blocks = nn.ModuleList(
            [DGABlock10Softplus(channels=int(self.image_encoder.embed_dim)) for _ in self.dga_indexes]
        )
        if record_intermediate_stats:
            attach_requested_intermediate_stats(
                self,
                record_intermediate_modules,
                {
                    "dga": [
                        (block, f"dga/block_{index}")
                        for index, block in enumerate(self.dga_blocks)
                    ],
                },
            )


class UNetFormerDGA20Softplus(UNetFormerDGA20):
    def __init__(
        self,
        *args: object,
        record_intermediate_stats: bool = False,
        record_intermediate_modules: Iterable[str] = (),
        **kwargs: object,
    ) -> None:
        super().__init__(
            *args,
            record_intermediate_stats=False,
            record_intermediate_modules=(),
            **kwargs,
        )
        self.dga_blocks = nn.ModuleList(
            [DGABlock20Softplus(channels=int(self.image_encoder.embed_dim)) for _ in self.dga_indexes]
        )
        if record_intermediate_stats:
            attach_requested_intermediate_stats(
                self,
                record_intermediate_modules,
                {
                    "dga": [
                        (block, f"dga/block_{index}")
                        for index, block in enumerate(self.dga_blocks)
                    ],
                },
            )


__all__ = ["UNetFormerDGA10Softplus", "UNetFormerDGA20Softplus"]
