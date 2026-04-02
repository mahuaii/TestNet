from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    def __init__(self, work_dir: str) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        trainer: Any,
    ) -> Path:
        path = self.work_dir / name
        torch.save(self._state_dict(
            model, optimizer, scheduler, trainer), path)
        torch.save(self._state_dict(model, optimizer, scheduler,
                   trainer), self.work_dir / "latest.pth")
        return path

    def resume(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        trainer: Any,
    ) -> dict[str, Any]:
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        if scheduler is not None and state_dict["scheduler"] is not None:
            scheduler.load_state_dict(state_dict["scheduler"])
        trainer.epoch = int(state_dict["epoch"])
        trainer.global_step = int(state_dict["global_step"])
        return state_dict

    def _state_dict(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        trainer: Any,
    ) -> dict[str, Any]:
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "epoch": trainer.epoch + 1,
            "global_step": trainer.global_step,
        }
