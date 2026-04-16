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
    ) -> dict[str, Any]:
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            # Persist the next epoch to run so training continues from the following epoch.
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_miou": float(best_miou),
        }
        return state_dict
