from __future__ import annotations

import torch
import torch.nn as nn

from .loss_modules import LossModule, ModelOutputs


class CombinedLoss(nn.Module):
    def __init__(self, losses: list[LossModule]) -> None:
        super().__init__()
        if not losses:
            raise ValueError("At least one loss must be configured.")
        self.losses = nn.ModuleList(losses)

    @property
    def class_weights(self) -> torch.Tensor | None:
        for loss_module in self.losses:
            class_weights = getattr(loss_module, "class_weights", None)
            if class_weights is not None:
                return class_weights
        return None

    def forward(
        self,
        outputs: ModelOutputs,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = LossModule.segmentation_logits(outputs)
        total = logits.sum() * 0.0
        items: dict[str, torch.Tensor] = {}

        for loss_module in self.losses:
            weighted_loss, loss_items = loss_module(outputs, target)
            duplicate_names = items.keys() & loss_items.keys()
            if duplicate_names:
                duplicates = ", ".join(sorted(duplicate_names))
                raise ValueError(f"Duplicate loss metric name(s): {duplicates}.")
            total = total + weighted_loss
            items.update(loss_items)

        return total, items


__all__ = ["CombinedLoss"]
