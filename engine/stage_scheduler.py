from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from losses import build_loss


@dataclass(frozen=True)
class Stage:
    name: str
    start_epoch: int
    end_epoch: int
    freeze_modules: tuple[str, ...]
    loss: list[str]
    loss_weights: Mapping[str, float] | None


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
            stage_names = ", ".join(stage.name for stage in matches)
            raise ValueError(
                f"Multiple training stages match epoch {epoch}: {stage_names}."
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

        if "name" not in raw_stage:
            raise KeyError(f"Stage {index} must define a name.")
        name = raw_stage["name"]
        if not isinstance(name, str) or not name:
            raise TypeError(f"Stage {index} name must be a non-empty string.")

        start_epoch, end_epoch = cls._parse_epochs(index=index, raw_stage=raw_stage)

        freeze_modules = raw_stage.get("freeze_modules")
        if not isinstance(freeze_modules, list) or not all(
            isinstance(path, str) and path for path in freeze_modules
        ):
            raise TypeError(
                f"Stage {name!r} freeze_modules must be a list of non-empty strings."
            )

        loss = raw_stage.get("loss")
        if not isinstance(loss, list) or not all(isinstance(name, str) for name in loss):
            raise TypeError(f"Stage {name!r} loss must be a list of strings.")
        if not loss:
            raise ValueError(f"Stage {name!r} loss must contain at least one item.")

        loss_weights = raw_stage.get("loss_weights")
        if loss_weights is not None and not isinstance(loss_weights, Mapping):
            raise TypeError(f"Stage {name!r} loss_weights must be a mapping.")

        return Stage(
            name=name,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            freeze_modules=tuple(freeze_modules),
            loss=list(loss),
            loss_weights=loss_weights,
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


__all__ = ["Stage", "StageScheduler"]
