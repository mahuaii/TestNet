from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from losses import build_loss
from utils import LR_SCOPE_DEFAULT, normalize_legacy_module_path


@dataclass(frozen=True)
class Stage:
    index: int
    start_epoch: int
    end_epoch: int
    freeze_modules: tuple[str, ...]
    loss: list[str]
    loss_weights: Mapping[str, float] | None
    default_lr: float | None
    module_lrs: Mapping[str, float]

    @property
    def has_lr_policy(self) -> bool:
        return self.default_lr is not None or bool(self.module_lrs)


class StageScheduler:
    def __init__(self, model: torch.nn.Module, stages: Any) -> None:
        if not isinstance(stages, list):
            raise TypeError("Stage configuration must be a list.")
        if not stages:
            raise ValueError("Stage configuration must contain at least one stage.")

        self._baseline_requires_grad = {
            name: param.requires_grad
            for name, param in model.named_parameters()
        }
        self._stages = tuple(
            self._parse_stage(index=index, raw_stage=raw_stage)
            for index, raw_stage in enumerate(stages)
        )
        self._validate_module_lr_paths(model)

    def apply(self, trainer: Any) -> None:
        stage = self.resolve_stage(int(trainer.epoch))
        self._restore_baseline(trainer.model)
        self._freeze_modules(trainer.model, stage.freeze_modules)

        trainer.criterion = build_loss(
            stage.loss,
            weights=stage.loss_weights,
            class_weights=trainer.cfg.get("class_weights"),
        ).to(trainer.device)
        trainer.class_weights = trainer.criterion.class_weights
        if stage.has_lr_policy:
            self._apply_stage_lrs(trainer, stage)

    def resolve_stage(self, epoch: int) -> Stage:
        if epoch < 1:
            raise ValueError(f"Epoch must be 1-based and positive, got {epoch}.")

        matches = [
            stage
            for stage in self._stages
            if stage.start_epoch <= epoch <= stage.end_epoch
        ]
        if not matches:
            raise ValueError(f"No training stage is configured for epoch {epoch}.")
        if len(matches) > 1:
            stage_indexes = ", ".join(str(stage.index) for stage in matches)
            raise ValueError(
                f"Multiple training stages match epoch {epoch}: indexes {stage_indexes}."
            )
        return matches[0]

    def _restore_baseline(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            param.requires_grad = self._baseline_requires_grad[name]

    @staticmethod
    def _freeze_modules(
        model: torch.nn.Module,
        module_paths: Sequence[str],
    ) -> None:
        for module_path in module_paths:
            module = model.get_submodule(module_path)
            for param in module.parameters():
                param.requires_grad = False

    @classmethod
    def _parse_stage(cls, *, index: int, raw_stage: Any) -> Stage:
        if not isinstance(raw_stage, Mapping):
            raise TypeError(f"Stage {index} must be a mapping.")

        start_epoch, end_epoch = cls._parse_epochs(index=index, raw_stage=raw_stage)

        freeze_modules = raw_stage.get("freeze_modules")
        if not isinstance(freeze_modules, list) or not all(
            isinstance(path, str) and path for path in freeze_modules
        ):
            raise TypeError(
                f"Stage {index} freeze_modules must be a list of non-empty strings."
            )

        loss = raw_stage.get("loss")
        if not isinstance(loss, list) or not all(isinstance(loss_name, str) for loss_name in loss):
            raise TypeError(f"Stage {index} loss must be a list of strings.")
        if not loss:
            raise ValueError(f"Stage {index} loss must contain at least one item.")

        loss_weights = raw_stage.get("loss_weights")
        if loss_weights is not None and not isinstance(loss_weights, Mapping):
            raise TypeError(f"Stage {index} loss_weights must be a mapping.")

        return Stage(
            index=index,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            freeze_modules=tuple(normalize_legacy_module_path(path) for path in freeze_modules),
            loss=list(loss),
            loss_weights=loss_weights,
            default_lr=cls._parse_optional_lr(
                stage_index=index,
                key="default_lr",
                value=raw_stage.get("default_lr"),
            ),
            module_lrs=cls._parse_module_lrs(stage_index=index, raw_stage=raw_stage),
        )

    @staticmethod
    def _parse_epochs(*, index: int, raw_stage: Mapping[str, Any]) -> tuple[int, int]:
        epochs = raw_stage.get("epochs")
        if not isinstance(epochs, list) or len(epochs) != 2:
            raise TypeError(f"Stage {index} epochs must be [start_epoch, end_epoch].")

        start_epoch, end_epoch = epochs
        if (
            isinstance(start_epoch, bool)
            or isinstance(end_epoch, bool)
            or not isinstance(start_epoch, int)
            or not isinstance(end_epoch, int)
        ):
            raise TypeError(f"Stage {index} epochs must contain integer values.")
        if start_epoch < 1:
            raise ValueError(f"Stage {index} start epoch must be positive.")
        if end_epoch < start_epoch:
            raise ValueError(
                f"Stage {index} end epoch must be greater than or equal to start epoch."
            )
        return start_epoch, end_epoch

    def _validate_module_lr_paths(self, model: torch.nn.Module) -> None:
        for stage in self._stages:
            for module_path in stage.module_lrs:
                model.get_submodule(module_path)

    @staticmethod
    def _apply_stage_lrs(trainer: Any, stage: Stage) -> None:
        optimizer = trainer.optimizer
        scheduler_scale = StageScheduler._scheduler_scale(trainer.scheduler)
        matched_module_lrs: set[str] = set()

        for group in optimizer.param_groups:
            lr_scope = str(group.get("lr_scope", LR_SCOPE_DEFAULT))
            nominal_lr = stage.module_lrs.get(lr_scope)
            if nominal_lr is not None:
                matched_module_lrs.add(lr_scope)
            elif stage.default_lr is not None:
                nominal_lr = stage.default_lr
            else:
                continue

            group["nominal_lr"] = nominal_lr
            group["lr"] = nominal_lr * scheduler_scale

        missing_module_lrs = sorted(set(stage.module_lrs) - matched_module_lrs)
        if missing_module_lrs:
            joined = ", ".join(missing_module_lrs)
            raise ValueError(f"Stage {stage.index} module_lrs did not match optimizer groups: {joined}.")

    @staticmethod
    def _scheduler_scale(scheduler: Any) -> float:
        if scheduler is None:
            return 1.0
        if not isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR):
            raise TypeError(
                "Stage lr policies currently support only torch.optim.lr_scheduler.MultiStepLR."
            )

        elapsed_milestones = sum(
            count
            for milestone, count in scheduler.milestones.items()
            if int(milestone) <= int(scheduler.last_epoch)
        )
        return float(scheduler.gamma) ** elapsed_milestones

    @classmethod
    def _parse_module_lrs(
        cls,
        *,
        stage_index: int,
        raw_stage: Mapping[str, Any],
    ) -> Mapping[str, float]:
        raw_module_lrs = raw_stage.get("module_lrs", {})
        if not isinstance(raw_module_lrs, Mapping):
            raise TypeError(f"Stage {stage_index} module_lrs must be a mapping.")

        module_lrs: dict[str, float] = {}
        for module_path, lr_value in raw_module_lrs.items():
            if not isinstance(module_path, str) or not module_path:
                raise TypeError(f"Stage {stage_index} module_lrs keys must be non-empty strings.")
            normalized_module_path = normalize_legacy_module_path(module_path)
            if normalized_module_path in module_lrs:
                raise ValueError(f"Stage {stage_index} has duplicate module_lrs path: {normalized_module_path}.")
            module_lrs[normalized_module_path] = cls._parse_required_lr(
                stage_index=stage_index,
                key=f"module_lrs[{module_path!r}]",
                value=lr_value,
            )
        return module_lrs

    @classmethod
    def _parse_optional_lr(cls, *, stage_index: int, key: str, value: Any) -> float | None:
        if value is None:
            return None
        return cls._parse_required_lr(stage_index=stage_index, key=key, value=value)

    @staticmethod
    def _parse_required_lr(*, stage_index: int, key: str, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Stage {stage_index} {key} must be a positive number.")
        lr = float(value)
        if lr <= 0:
            raise ValueError(f"Stage {stage_index} {key} must be positive, got {lr}.")
        return lr


__all__ = ["Stage", "StageScheduler"]
