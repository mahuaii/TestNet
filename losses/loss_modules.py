from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from utils import DataUtils

from .boundary_loss import boundary_loss
from .lovasz_softmax import lovasz_softmax_loss

ModelOutputs = torch.Tensor | Mapping[str, torch.Tensor]


class LossModule(nn.Module):
    name: str

    def __init__(self, *, weight: float) -> None:
        super().__init__()
        self.weight = float(weight)

    def forward(
        self,
        outputs: ModelOutputs,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    @staticmethod
    def segmentation_logits(outputs: ModelOutputs) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if "logits" not in outputs:
            raise KeyError("Model output mapping must contain 'logits'.")
        return outputs["logits"]


class CrossEntropyLossModule(LossModule):
    name = "ce"

    def __init__(
        self,
        *,
        class_weights: list[float] | None = None,
        ignore_index: int = 255,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight=weight)
        self.ignore_index = int(ignore_index)
        weights = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.register_buffer("class_weights", weights)

    def forward(
        self,
        outputs: ModelOutputs,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss = DataUtils.cross_entropy_filtered(
            logits=self.segmentation_logits(outputs),
            target=target,
            weight=self.class_weights,
            ignore_label=self.ignore_index,
        )
        return self.weight * loss, {self.name: loss}


class LovaszLossModule(LossModule):
    name = "lovasz"

    def __init__(
        self,
        *,
        ignore_index: int = 255,
        classes: str = "present",
        weight: float = 0.2,
    ) -> None:
        super().__init__(weight=weight)
        self.ignore_index = int(ignore_index)
        self.classes = str(classes)

    def forward(
        self,
        outputs: ModelOutputs,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        probabilities = torch.softmax(self.segmentation_logits(outputs), dim=1)
        loss = lovasz_softmax_loss(
            probabilities,
            target,
            classes=self.classes,
            ignore_index=self.ignore_index,
        )
        return self.weight * loss, {self.name: loss}


class BoundaryLossModule(LossModule):
    name = "boundary"

    def __init__(
        self,
        *,
        ignore_index: int = 255,
        pos_weight: float = 3.0,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_width: int = 1,
        weight: float = 0.05,
    ) -> None:
        super().__init__(weight=weight)
        self.ignore_index = int(ignore_index)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.boundary_width = int(boundary_width)
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32),
        )

    def forward(
        self,
        outputs: ModelOutputs,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(outputs, torch.Tensor) or "boundary_logits" not in outputs:
            raise KeyError(
                "Boundary loss is enabled, but model output does not contain "
                "'boundary_logits'."
            )
        loss, items = boundary_loss(
            outputs["boundary_logits"],
            target,
            ignore_index=self.ignore_index,
            pos_weight=self.pos_weight,
            bce_weight=self.bce_weight,
            dice_weight=self.dice_weight,
            boundary_width=self.boundary_width,
        )
        return self.weight * loss, {self.name: loss, **items}


__all__ = [
    "BoundaryLossModule",
    "CrossEntropyLossModule",
    "LossModule",
    "LovaszLossModule",
    "ModelOutputs",
]
