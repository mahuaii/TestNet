from __future__ import annotations

from typing import Any

import torch
from typing_extensions import override

from .mfnet_dga_trainer import MFNetDGATrainer


class MFNetDGAContributionStatsTrainer(MFNetDGATrainer):
    @override
    def train_forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        loss, metrics = super().train_forward(batch)
        metrics.update(self._collect_dga_contribution_scalars())
        return loss, metrics

    def _collect_dga_contribution_scalars(self) -> dict[str, float]:
        dga_blocks = getattr(self.model, "dga_blocks", None)
        if dga_blocks is None:
            return {}

        scalars: dict[str, float] = {}
        for index, block in enumerate(dga_blocks):
            stats = getattr(block, "last_dga_stats", None)
            if not stats:
                continue
            for name, value in stats.items():
                scalars[f"dga/{name}_block_{index}"] = self._to_float(value)
        return scalars


__all__ = ["MFNetDGAContributionStatsTrainer"]
