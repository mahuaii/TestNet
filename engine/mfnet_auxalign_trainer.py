from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from typing_extensions import override

from .evaluator import Evaluator
from .mfnet_trainer import MFNetTrainer
from utils import DataUtils
from utils.stat_tracker import StatTracker


class MFNetAuxAlignTrainer(MFNetTrainer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_align = float(self.cfg.get("lambda_align", 0.01))

    @override
    def train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        self.timer.mark("epoch")
        self.timer.mark("log_interval")

        step = 0
        epoch_metrics = StatTracker()
        log_window_metrics = StatTracker()
        self.optimizer.zero_grad()

        for batch in self.train_loader:
            _, metrics = self.train_forward(batch)

            log_window_metrics.update_mean_stats(metrics)
            epoch_metrics.update_mean_stats(metrics)
            self.optimize_step()
            step += 1
            self.after_step(
                step,
                log_window_metrics,
                is_last_step_of_epoch=(step == len(self.train_loader)),
            )
            if step % int(self.cfg["log_step_interval"]) == 0 or step == len(self.train_loader):
                log_window_metrics = StatTracker()

        return epoch_metrics.get_aggregated_stats()

    @override
    def train_forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        rgb, dsm, target = self._extract_train_tensors(batch)
        output = self.model(rgb, dsm, mode="Train", return_align=True)
        if not isinstance(output, tuple) or len(output) != 3:
            raise TypeError(
                "MFNetAuxAlignTrainer expects the model to return "
                "(logits, x_align_feat, y_align_feat) when return_align=True."
            )
        logits, x_align_feat, y_align_feat = output

        loss_seg = DataUtils.cross_entropy_filtered(
            logits=logits,
            target=target,
            weight=self.class_weights,
        )
        loss_align = F.mse_loss(y_align_feat, x_align_feat.detach())

        prealign_params = [
            param
            for param in self.model.aux_prealign.parameters()
            if param.requires_grad
        ]
        if not prealign_params:
            raise ValueError("No trainable aux_prealign parameters were found.")

        align_grads = torch.autograd.grad(
            self.lambda_align * loss_align,
            prealign_params,
            retain_graph=True,
            allow_unused=False,
        )
        loss_seg.backward()
        for param, grad in zip(prealign_params, align_grads):
            if param.grad is None:
                param.grad = grad.detach().clone()
            else:
                param.grad.add_(grad)

        loss = loss_seg.detach() + self.lambda_align * loss_align.detach()
        pred = torch.argmax(logits.detach(), dim=1)
        accuracy = Evaluator.accuracy(pred=pred, target=target)
        metrics = {
            "loss": float(loss),
            "loss_seg": float(loss_seg.detach()),
            "loss_align": float(loss_align.detach()),
            "accuracy": accuracy,
        }
        return loss, metrics
