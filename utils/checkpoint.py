from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    def __init__(self, work_dir: str) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def save_training_state(
        self,
        name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        global_step: int,
    ) -> Path:
        path = self.work_dir / name
        state_dict = self._state_dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
        )
        torch.save(state_dict, path)
        torch.save(state_dict, self.work_dir / "latest.pth")
        return path

    def load(self, path: str) -> dict[str, Any]:
        state_dict = torch.load(path, map_location="cpu")
        return state_dict

    def _state_dict(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        global_step: int,
    ) -> dict[str, Any]:
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            # Persist the next epoch to run so training continues from the following epoch.
            "epoch": epoch + 1,
            "global_step": global_step,
        }
        return state_dict
