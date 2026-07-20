from __future__ import annotations

from typing import Any

import torch
from typing_extensions import override

from utils.stat_tracker import StatTracker

from .mfnet_trainer import MFNetTrainer


class MFNetDGATrainer(MFNetTrainer):
    @override
    def after_step(
        self,
        step: int,
        step_stats_tracker: StatTracker,
        is_last_step_of_epoch: bool = False,
    ) -> None:
        if step % int(self.cfg["log_step_interval"]) == 0 or is_last_step_of_epoch:
            step_stats = step_stats_tracker.get_aggregated_stats()
            step_stats.update(self._collect_dga_block_scalars())
            self.logger.log_train_step(
                epoch=self.epoch,
                max_epochs=self.max_epochs,
                step=step,
                total_steps=self.total_steps_per_epoch,
                global_step=self.global_step,
                step_stats=step_stats,
                interval_time_seconds=self.timer.elapsed("log_interval"),
                epoch_elapsed_seconds=self.timer.elapsed("epoch"),
                lr=self.lr,
            )
            self.logger.write_train_step_scalars(
                global_step=self.global_step,
                step_stats=step_stats,
                lr=self.lr,
            )
            self.timer.mark("log_interval")

        save_step_interval = self.cfg["save_step_interval"]
        if save_step_interval > 0 and self.global_step % save_step_interval == 0:
            self._save_training_state(
                name=f"global_step_{self.global_step}.pth",
                resume_epoch=self.epoch,
            )

    def _collect_dga_block_scalars(self) -> dict[str, float]:
        dga_blocks = getattr(self.model, "dga_blocks", None)
        if dga_blocks is None:
            return {}

        scalars: dict[str, float] = {}
        for index, block in enumerate(dga_blocks):
            alpha = self._gate_scalar(block, parameter_name="alpha", effective_method_name="effective_alpha")
            beta = self._gate_scalar(block, parameter_name="beta", effective_method_name="effective_beta")
            if alpha is not None:
                scalars[f"dga/alpha_block_{index}"] = self._to_float(alpha)
            if beta is not None:
                scalars[f"dga/beta_block_{index}"] = self._to_float(beta)
        return scalars

    @staticmethod
    def _gate_scalar(block: Any, *, parameter_name: str, effective_method_name: str) -> Any:
        effective_method = getattr(block, effective_method_name, None)
        if callable(effective_method):
            return effective_method()
        return getattr(block, parameter_name, None)

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)


__all__ = ["MFNetDGATrainer"]
