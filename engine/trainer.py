from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from .evaluator import Evaluator
from utils.checkpoint import CheckpointManager
from utils.logger import Logger
from utils.meter import RunningMetricTracker


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: Any,
        val_loader: Any,
        logger: Logger,
        checkpoint_manager: CheckpointManager,
        evaluator: Evaluator,
        device: torch.device,
        cfg: dict[str, Any],
        scheduler: Any = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.evaluator = evaluator
        self.device = device
        self.cfg = cfg

        self.epoch = 0
        self.global_step = 0
        self.max_epochs = int(cfg["max_epochs"])

    def train(self) -> None:
        resume_from = self.cfg.get("resume_from")
        load_from = self.cfg.get("load_from")
        if load_from:
            state_dict = torch.load(load_from, map_location="cpu")
            self.model.load_state_dict(
                state_dict["model"] if "model" in state_dict else state_dict)
        if resume_from:
            self.checkpoint_manager.resume(
                resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                trainer=self,
            )

        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            train_metrics, train_state_metrics = self.train_one_epoch()
            if (epoch + 1) % int(self.cfg["val_interval"]) == 0:
                val_metrics = self.validate()
            else:
                val_metrics = None
            self.after_epoch(train_metrics, train_state_metrics, val_metrics)

    def train_one_epoch(self) -> tuple[dict[str, float], dict[str, float]]:
        self.model.train()
        running_metric_tracker = RunningMetricTracker()
        train_state_metrics = {"lr": self.optimizer.param_groups[0]["lr"]}
        end = time.time()

        for iter_idx, batch in enumerate(self.train_loader, start=1):
            running_metric_tracker.update(data_time=time.time() - end)
            iter_start = time.time()
            metrics = self.train_step(batch)
            self.global_step += 1
            running_metric_tracker.update(
                **metrics, iter_time=time.time() - iter_start)
            train_state_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            self.after_iter(iter_idx, running_metric_tracker,
                            train_state_metrics)
            end = time.time()

        return running_metric_tracker.get_average_metrics(), dict(train_state_metrics)

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        inputs = {k: v.to(self.device) for k, v in batch["inputs"].items()}
        target = batch["target"].to(self.device)

        outputs = self.model(inputs, mode="loss")
        loss = self._compute_loss(outputs["seg_logits"], target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return {"loss": float(loss.item())}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        outputs = []
        for batch in self.val_loader:
            outputs.append(self.val_step(batch))
        return self.evaluator.evaluate(outputs, num_classes=int(self.cfg["num_classes"]))

    @torch.no_grad()
    def val_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | float]:
        inputs = {k: v.to(self.device) for k, v in batch["inputs"].items()}
        target = batch["target"].to(self.device)
        outputs = self.model(inputs, mode="predict")
        loss = self._compute_loss(outputs["seg_logits"], target)
        return {
            "seg_logits": outputs["seg_logits"].detach().cpu(),
            "target": target.detach().cpu(),
            "loss": float(loss.item()),
        }

    def after_iter(
        self,
        iter_idx: int,
        running_metric_tracker: RunningMetricTracker,
        train_state_metrics: dict[str, float],
    ) -> None:
        if iter_idx % int(self.cfg["log_interval"]) == 0:
            self.logger.log_iter(
                epoch=self.epoch + 1,
                iter_idx=iter_idx,
                global_step=self.global_step,
                running_metrics=running_metric_tracker.get_latest_metrics(),
                state_metrics=train_state_metrics,
            )
        save_iter_interval = int(self.cfg["save_iter_interval"])
        if save_iter_interval > 0 and self.global_step % save_iter_interval == 0:
            self.checkpoint_manager.save(
                name=f"step_{self.global_step}.pth",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                trainer=self,
            )

    def after_epoch(
        self,
        train_metrics: dict[str, float],
        train_state_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        self.logger.log_epoch(
            epoch=self.epoch + 1,
            train_metrics=train_metrics,
            train_state_metrics=train_state_metrics,
            val_metrics=val_metrics,
        )

        if (self.epoch + 1) % int(self.cfg["save_epoch_interval"]) == 0:
            self.checkpoint_manager.save(
                name=f"epoch_{self.epoch + 1}.pth",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                trainer=self,
            )

    def _compute_loss(self, seg_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if int(self.cfg["num_classes"]) == 1:
            return F.binary_cross_entropy_with_logits(seg_logits.squeeze(1), target.float())
        return F.cross_entropy(seg_logits, target.long())
