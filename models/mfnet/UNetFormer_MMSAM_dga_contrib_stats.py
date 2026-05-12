from __future__ import annotations

import torch.nn as nn

from .UNetFormer_MMSAM_dga10 import UNetFormerDGA10
from .UNetFormer_MMSAM_dga20 import UNetFormerDGA20
from .modules.dga_contrib_stats import (
    DGABlock10ContributionStats,
    DGABlock10ContributionStatsSoftplus,
    DGABlock20ContributionStats,
    DGABlock20ContributionStatsSoftplus,
)


class UNetFormerDGA10ContributionStats(UNetFormerDGA10):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.dga_blocks = nn.ModuleList(
            [DGABlock10ContributionStats(channels=int(self.image_encoder.embed_dim)) for _ in self.dga_indexes]
        )


class UNetFormerDGA20ContributionStats(UNetFormerDGA20):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.dga_blocks = nn.ModuleList(
            [DGABlock20ContributionStats(channels=int(self.image_encoder.embed_dim)) for _ in self.dga_indexes]
        )


class UNetFormerDGA10ContributionStatsSoftplus(UNetFormerDGA10ContributionStats):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.dga_blocks = nn.ModuleList(
            [DGABlock10ContributionStatsSoftplus(channels=int(self.image_encoder.embed_dim)) for _ in self.dga_indexes]
        )


class UNetFormerDGA20ContributionStatsSoftplus(UNetFormerDGA20ContributionStats):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.dga_blocks = nn.ModuleList(
            [DGABlock20ContributionStatsSoftplus(channels=int(self.image_encoder.embed_dim)) for _ in self.dga_indexes]
        )


__all__ = [
    "UNetFormerDGA10ContributionStats",
    "UNetFormerDGA10ContributionStatsSoftplus",
    "UNetFormerDGA20ContributionStats",
    "UNetFormerDGA20ContributionStatsSoftplus",
]
