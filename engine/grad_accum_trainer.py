from __future__ import annotations

import math

from utils.stat_tracker import StatTracker
from typing_extensions import override

from .trainer import Trainer


class GradAccumTrainer(Trainer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)

        effective_batch_size = self.cfg.get("effective_batch_size", self.train_loader.batch_size)
        if effective_batch_size < self.train_loader.batch_size:
            raise ValueError("effective_batch_size must be greater than or equal to batch_size")
        if effective_batch_size % self.train_loader.batch_size != 0:
            raise ValueError("effective_batch_size must be divisible by batch_size")
        self.grad_accum_steps = effective_batch_size // self.train_loader.batch_size
        self.total_steps_per_epoch = max(
            1, math.ceil(len(self.train_loader) / self.grad_accum_steps)
        )

    @override
    def train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        self.timer.mark("epoch")
        self.timer.mark("log_interval")

        step = 0
        batch_count_in_step = 0
        batch_count_in_epoch = 0
        epoch_metrics = StatTracker()
        total_batches = len(self.train_loader)

        self.optimizer.zero_grad()

        log_window_metrics = StatTracker()
        for batch in self.train_loader:
            remaining_batches = total_batches - batch_count_in_epoch
            accum_batch_target = min(self.grad_accum_steps, remaining_batches)

            loss, metrics = self.train_forward(batch)
            (loss / accum_batch_target).backward()

            log_window_metrics.update_mean_stats(metrics)
            epoch_metrics.update_mean_stats(metrics)
            batch_count_in_step += 1
            batch_count_in_epoch += 1

            if (batch_count_in_step == self.grad_accum_steps) or (
                batch_count_in_epoch == total_batches
            ):
                self.optimize_step()
                step += 1
                self.after_step(
                    step,
                    log_window_metrics,
                    is_last_step_of_epoch=(batch_count_in_epoch == total_batches),
                )
                batch_count_in_step = 0
                if step % int(self.cfg["log_step_interval"]) == 0 or (
                    batch_count_in_epoch == total_batches
                ):
                    log_window_metrics = StatTracker()

        return epoch_metrics.get_aggregated_stats()
