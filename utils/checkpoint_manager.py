from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    @staticmethod
    def save_training_state(
        path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        global_step: int,
        best_miou: float,
        resume_epoch: int | None = None,
        validation_pending: bool = False,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = CheckpointManager._state_dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_miou=best_miou,
            resume_epoch=resume_epoch,
            validation_pending=validation_pending,
        )
        torch.save(state_dict, path)
        return path

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        state_dict = torch.load(path, map_location="cpu")
        return state_dict

    @staticmethod
    def _state_dict(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        global_step: int,
        best_miou: float,
        resume_epoch: int | None = None,
        validation_pending: bool = False,
    ) -> dict[str, Any]:
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "epoch": epoch,
            "resume_epoch": epoch if resume_epoch is None else int(resume_epoch),
            "global_step": global_step,
            "best_miou": float(best_miou),
            "validation_pending": bool(validation_pending),
        }
        return state_dict
