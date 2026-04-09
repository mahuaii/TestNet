from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from typing_extensions import override

from .evaluator import Evaluator
from .trainer import Trainer


class MFNetTrainer(Trainer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        class_weights = self.cfg.get("class_weights")
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            if class_weights is not None
            else None
        )

    @override
    def before_epoch(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    @override
    def train_forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        rgb, dsm, target = self._extract_train_tensors(batch)
        self._validate_train_tensors(rgb=rgb, dsm=dsm, target=target)
        logits = self.model(rgb, dsm, mode="Train")
        loss, metrics = self._compute_loss_and_metrics(logits=logits, target=target)
        return loss, metrics

    def _extract_train_tensors(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = batch["inputs"]
        rgb = inputs["rgb"].to(self.device, non_blocking=True)
        dsm = inputs["dsm"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        return rgb, dsm, target

    def _validate_train_tensors(
        self,
        rgb: torch.Tensor,
        dsm: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        if rgb.ndim != 4:
            raise ValueError(
                f"MFNetTrainer expected RGB with shape [B, 3, H, W], got {tuple(rgb.shape)}"
            )
        if dsm.ndim != 3:
            raise ValueError(
                f"MFNetTrainer expected DSM with shape [B, H, W], got {tuple(dsm.shape)}"
            )
        if target.ndim != 3:
            raise ValueError(
                f"MFNetTrainer expected target with shape [B, H, W], got {tuple(target.shape)}"
            )
        if target.dtype != torch.long:
            raise TypeError(f"MFNetTrainer expected target dtype torch.long, got {target.dtype}")

    def _compute_loss_and_metrics(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
        )
        pred = torch.argmax(logits.detach(), dim=1)
        accuracy = Evaluator.accuracy(pred=pred, target=target)
        return loss, {"loss": float(loss.detach()), "accuracy": accuracy}
