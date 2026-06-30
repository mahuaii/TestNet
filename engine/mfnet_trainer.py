from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from typing_extensions import override

from losses import CombinedLoss, build_loss

from .evaluator import Evaluator
from .stage_scheduler import StageScheduler
from .training_diagnostics import collect_optimizer_group_summaries, stage_label

from .trainer import Trainer


class MFNetTrainer(Trainer):
    def __init__(
        self,
        *args: object,
        criterion: CombinedLoss | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.criterion = (
            criterion
            if criterion is not None
            else build_loss(
                self.cfg.get("loss"),
                weights=self.cfg.get("loss_weights"),
                class_weights=self.cfg.get("class_weights"),
            )
        )
        self.criterion = self.criterion.to(self.device)
        self.class_weights = self.criterion.class_weights
        self.stage_scheduler = (
            StageScheduler(self.model, self.cfg["stages"])
            if "stages" in self.cfg
            else None
        )
        self.current_stage = None

    @override
    def before_epoch(self) -> None:
        super().before_epoch()
        if self.scheduler is not None:
            self.scheduler.step()
        self.current_stage = None
        if self.stage_scheduler is not None:
            self.current_stage = self.stage_scheduler.apply(self)

    @override
    def after_epoch_start_logged(self) -> None:
        current_stage_label = stage_label(self.current_stage)
        self.logger.log_lr_groups(
            epoch=self.epoch,
            max_epochs=self.max_epochs,
            stage_label=current_stage_label,
            scheduler_scale=StageScheduler._scheduler_scale(self.scheduler),
            group_summaries=collect_optimizer_group_summaries(self.optimizer),
        )

    @override
    def train_forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        rgb, dsm, target = self._extract_train_tensors(batch)
        outputs = self.model(rgb, dsm, mode="Train")
        loss, metrics = self._compute_loss_and_metrics(logits=outputs, target=target)
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
        logits: torch.Tensor | Mapping[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        segmentation_logits = self._extract_segmentation_logits(logits)
        loss, loss_items = self._compute_segmentation_loss(outputs=logits, target=target)
        pred = torch.argmax(segmentation_logits.detach(), dim=1)
        accuracy = Evaluator.accuracy(pred=pred, target=target)
        metrics = {
            "loss": float(loss.detach()),
            "accuracy": accuracy,
        }
        metrics.update(
            {
                f"loss_{name}": float(value.detach())
                for name, value in loss_items.items()
            }
        )
        return loss, metrics

    def _compute_segmentation_loss(
        self,
        *,
        outputs: torch.Tensor | Mapping[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.criterion(outputs, target)

    @staticmethod
    def _extract_segmentation_logits(
        outputs: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if "logits" not in outputs:
            raise KeyError("Model output mapping must contain 'logits'.")
        return outputs["logits"]
