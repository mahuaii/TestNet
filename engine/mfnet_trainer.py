from __future__ import annotations

from typing import Any

import torch
from typing_extensions import override

from .evaluator import Evaluator
from utils import DataUtils

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
        intermediate_stats = getattr(self.model, "intermediate_stats", None)
        if intermediate_stats is not None:
            intermediate_stats.clear()
        logits = self.model(rgb, dsm, mode="Train")
        loss, metrics = self._compute_loss_and_metrics(logits=logits, target=target)
        if intermediate_stats is not None:
            metrics.update(intermediate_stats.snapshot(reset=True))
        return loss, metrics

    @override
    @torch.no_grad()
    def validate(self) -> None:
        self.timer.mark("validation")
        self.model.eval()

        validation_cfg = self.cfg["validation"]
        dataset = self.val_loader.dataset
        outputs = self.inferencer.run(
            model=self.model,
            dataset=dataset,
            device=self.device,
            stride=int(validation_cfg["stride"]),
            batch_size=int(self.cfg["batch_size"]),
            window_size=tuple(dataset.patch_size),
            num_classes=int(self.cfg["num_classes"]),
            input_modals=("rgb", "dsm"),
            model_kwargs={"mode": "Test"},
        )
        val_metrics = self.evaluator.evaluate(
            outputs=outputs,
            num_classes=int(self.cfg["num_classes"]),
        )
        validation_time_seconds = self.timer.elapsed("validation")
        self.after_val(
            val_metrics,
            validation_time_seconds=validation_time_seconds,
        )
        self.model.train()

    def _extract_train_tensors(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = batch["inputs"]
        rgb = inputs["rgb"].to(self.device, non_blocking=True)
        dsm = inputs["dsm"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        if dsm.ndim != 3:
            raise ValueError(f"expected DSM with shape [B, H, W], got {tuple(dsm.shape)}")
        if target.dtype != torch.long:
            raise TypeError(f"expected target dtype torch.long, got {target.dtype}")
        return rgb, dsm, target

    def _compute_loss_and_metrics(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss = DataUtils.cross_entropy_filtered(
            logits=logits,
            target=target,
            weight=self.class_weights,
        )
        pred = torch.argmax(logits.detach(), dim=1)
        accuracy = Evaluator.accuracy(pred=pred, target=target)
        return loss, {"loss": float(loss.detach()), "accuracy": accuracy}
