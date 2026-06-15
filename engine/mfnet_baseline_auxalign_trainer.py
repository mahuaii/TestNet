from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from typing_extensions import override

from .evaluator import Evaluator
from .mfnet_trainer import MFNetTrainer
from utils.stat_tracker import StatTracker


class MFNetBaselineAuxAlignTrainer(MFNetTrainer):
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
            _, metrics = self._run_train_forward(batch)

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
                "MFNetBaselineAuxAlignTrainer expects the model to return "
                "(logits, x_align_feat, y_align_feat) when return_align=True."
            )
        logits, x_align_feat, y_align_feat = output
        segmentation_logits = self._extract_segmentation_logits(logits)
        self._raise_if_nonfinite("logits", segmentation_logits)
        self._raise_if_nonfinite("x_align_feat", x_align_feat)
        self._raise_if_nonfinite("y_align_feat", y_align_feat)

        loss_seg, loss_items = self._compute_segmentation_loss(
            outputs=logits,
            target=target,
        )
        loss_align = F.mse_loss(y_align_feat, x_align_feat.detach())
        self._raise_if_nonfinite("loss_seg", loss_seg)
        self._raise_if_nonfinite("loss_align", loss_align)

        self._backward_scoped_losses(loss_seg=loss_seg, loss_align=loss_align)

        loss = loss_seg.detach() + self.lambda_align * loss_align.detach()
        pred = torch.argmax(segmentation_logits.detach(), dim=1)
        accuracy = Evaluator.accuracy(pred=pred, target=target)
        metrics = {
            "loss": float(loss),
            "loss_seg": float(loss_seg.detach()),
            "loss_align": float(loss_align.detach()),
            "accuracy": accuracy,
        }
        metrics.update(
            {
                f"loss_{name}": float(value.detach())
                for name, value in loss_items.items()
            }
        )
        return loss, metrics

    def _backward_scoped_losses(
        self,
        *,
        loss_seg: torch.Tensor,
        loss_align: torch.Tensor,
    ) -> None:
        aux_align_params = self._aux_align_adapter_params()
        if not aux_align_params:
            raise ValueError(
                "No trainable DSM/aux Adapter parameters were found before the align block."
            )

        align_grads = torch.autograd.grad(
            self.lambda_align * loss_align,
            aux_align_params,
            retain_graph=True,
            allow_unused=False,
        )
        loss_seg.backward()
        self._add_grads(aux_align_params, align_grads)

    def _aux_align_adapter_params(self) -> list[torch.nn.Parameter]:
        image_encoder = getattr(self.model, "image_encoder", None)
        blocks = getattr(image_encoder, "blocks", None)
        align_index = getattr(self.model, "align_index", None)
        if blocks is None or align_index is None:
            return []

        params: list[torch.nn.Parameter] = []
        for block in list(blocks)[: int(align_index) + 1]:
            for name, param in block.named_parameters():
                if not param.requires_grad:
                    continue
                if "DSM_Adapter" in name or "MLPy_Adapter" in name:
                    params.append(param)
        return params

    @staticmethod
    def _add_grads(
        params: list[torch.nn.Parameter],
        grads: tuple[torch.Tensor, ...],
    ) -> None:
        for param, grad in zip(params, grads):
            if param.grad is None:
                param.grad = grad.detach().clone()
            else:
                param.grad.add_(grad)

    @staticmethod
    def _raise_if_nonfinite(name: str, tensor: torch.Tensor) -> None:
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(
                f"MFNetBaselineAuxAlignTrainer encountered non-finite {name}."
            )


__all__ = ["MFNetBaselineAuxAlignTrainer"]
