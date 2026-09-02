from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from typing_extensions import override

from utils.metric_accumulator_3090 import DeviceMetricAccumulator
from utils.stat_tracker import StatTracker

from .mfnet_trainer import MFNetTrainer


class MFNet3090Trainer(MFNetTrainer):
    """BF16 trainer for the SPMF + Prealign model family."""

    _SUPPORTED_PRECISION = "bfloat16"

    def __init__(self, *args: object, **kwargs: object) -> None:
        cfg = kwargs.get("cfg")
        if not isinstance(cfg, Mapping):
            raise TypeError("MFNet3090Trainer requires a mapping cfg argument.")

        runtime_cfg = cfg.get("runtime_3090")
        if not isinstance(runtime_cfg, Mapping):
            raise KeyError("MFNet3090Trainer requires cfg.runtime_3090.")
        if runtime_cfg.get("precision") != self._SUPPORTED_PRECISION:
            raise ValueError(
                "MFNet3090Trainer supports only runtime_3090.precision='bfloat16'."
            )

        device = kwargs.get("device")
        if not isinstance(device, torch.device) or device.type != "cuda":
            raise ValueError("MFNet3090Trainer requires a CUDA device.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support BF16 autocast.")

        super().__init__(*args, **kwargs)
        self._amp_dtype = torch.bfloat16
        self._last_epoch_samples = 0

    @override
    def before_epoch(self) -> None:
        super().before_epoch()
        torch.cuda.reset_peak_memory_stats(self.device)

    def _autocast_context(self):
        return torch.autocast(
            device_type="cuda",
            dtype=self._amp_dtype,
            enabled=True,
        )

    @override
    def train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        self.timer.mark("epoch")
        self.timer.mark("log_interval")

        step = 0
        self._last_epoch_samples = 0
        epoch_metrics = DeviceMetricAccumulator()
        log_window_metrics = DeviceMetricAccumulator()
        self.optimizer.zero_grad()

        for batch in self.train_loader:
            self._last_epoch_samples += int(batch["target"].shape[0])
            with self._autocast_context():
                loss, metrics = self._run_train_forward_3090(batch)
            loss.backward()

            epoch_metrics.update(metrics)
            log_window_metrics.update(metrics)
            self.optimize_step()
            step += 1

            is_last_step = step == len(self.train_loader)
            should_log = step % int(self.cfg["log_step_interval"]) == 0 or is_last_step
            step_tracker = StatTracker()
            if should_log:
                step_tracker.update_mean_stats(log_window_metrics.snapshot())
            self.after_step(
                step,
                step_tracker,
                is_last_step_of_epoch=is_last_step,
            )
            if should_log:
                log_window_metrics.reset()

        return epoch_metrics.snapshot()

    @override
    def after_epoch(
        self,
        train_metrics: dict[str, float],
        train_time_seconds: float,
        validation_pending: bool = False,
    ) -> None:
        super().after_epoch(
            train_metrics,
            train_time_seconds=train_time_seconds,
            validation_pending=validation_pending,
        )
        samples_per_second = (
            self._last_epoch_samples / train_time_seconds
            if train_time_seconds > 0.0
            else 0.0
        )
        peak_memory_gib = torch.cuda.max_memory_allocated(self.device) / 2**30
        self.logger.log_message(
            "3090 performance | "
            f"samples/s: {samples_per_second:.3f} | "
            f"epoch train time: {train_time_seconds:.3f}s | "
            f"peak GPU memory: {peak_memory_gib:.3f} GiB"
        )

    @override
    def train_forward(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rgb, dsm, target = self._extract_train_tensors(batch)
        outputs = self.model(rgb, dsm, mode="Train")
        segmentation_logits = self._extract_segmentation_logits(outputs)
        loss, loss_items = self._compute_segmentation_loss(
            outputs=outputs,
            target=target,
        )
        pred = torch.argmax(segmentation_logits.detach(), dim=1)
        metrics: dict[str, torch.Tensor] = {
            "loss": loss.detach(),
            "accuracy_correct": (pred == target).to(torch.float32).sum(),
            "accuracy_total": torch.tensor(
                target.numel(),
                device=target.device,
                dtype=torch.float32,
            ),
        }
        metrics.update(
            {
                f"loss_{name}": value.detach()
                for name, value in loss_items.items()
            }
        )
        return loss, metrics

    def _run_train_forward_3090(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self.before_train_forward()
        loss, metrics = self.train_forward(batch)
        return loss, self.after_train_forward(metrics)  # type: ignore[arg-type]

    @override
    @torch.inference_mode()
    def validate(self) -> None:
        self.timer.mark("validation")
        self.model.eval()

        validation_cfg = self.cfg["validation"]
        dataset = self.val_loader.dataset
        validation_batch_size = int(
            self.cfg.get("val_batch_size", self.cfg["batch_size"])
        )
        with self._autocast_context():
            outputs = self.inferencer.run(
                model=self.model,
                dataset=dataset,
                device=self.device,
                stride=int(validation_cfg["stride"]),
                batch_size=validation_batch_size,
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


__all__ = ["MFNet3090Trainer"]
