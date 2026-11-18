"""Model-specific construction, training freeze policy and inference input adaptation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from models import build_model
from utils.train_utils import normalize_legacy_state_dict_keys, resolve_config_module_path


class EfficiencyAdapter(nn.Module):
    """Adapters own the whole model; conversions/resizing belong in ``forward``.

    A custom factory accepts ``(cfg, checkpoint, train_epoch)`` and returns an
    instance of this class. Canonical inputs are FP32 RGB BCHW and DSM BHW.
    Do not cache converted inputs or encoder features across calls.
    """

    def __init__(self, model: nn.Module, metadata: dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.metadata = metadata

    def forward(self, rgb: torch.Tensor, dsm: torch.Tensor) -> Any:
        raise NotImplementedError


class RepositoryAdapter(EfficiencyAdapter):
    """MFNet and ScSAM use the same existing repository inference entry."""

    def forward(self, rgb: torch.Tensor, dsm: torch.Tensor) -> torch.Tensor:
        return self.model(rgb, dsm, mode="Test")


def build_repository_adapter(
    cfg: dict[str, Any], checkpoint: Path | None, train_epoch: int | None
) -> EfficiencyAdapter:
    model_cfg = dict(cfg["model"])
    # Training diagnostics are not required for inference.
    model_cfg["record_intermediate_stats"] = False
    # Fail before the SAM constructor can offer an interactive download.
    model_cfg["sam_checkpoint"] = str(Path(model_cfg["sam_checkpoint"]).expanduser().resolve(strict=True))
    model = build_model(model_cfg)
    checkpoint_epoch = None
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # Explicitly configurable for future checkpoints; null means raw state_dict.
        key = cfg.get("efficiency", {}).get("checkpoint_key", "model")
        weights = state if key is None else state[key]
        model.load_state_dict(normalize_legacy_state_dict_keys(weights), strict=True)
        if key is not None and "epoch" in state:
            checkpoint_epoch = int(state["epoch"])
        del weights, state

    epoch = train_epoch if train_epoch is not None else checkpoint_epoch
    metadata: dict[str, Any] = {
        "model_type": model_cfg["type"],
        "model_config": model_cfg,
        "task_weights": "checkpoint" if checkpoint is not None else "initialization",
        "checkpoint_epoch": checkpoint_epoch,
        "train_epoch": epoch,
        "freeze_policy": "constructor",
        "input_adapter": "model(rgb_BCHW, dsm_BHW, mode='Test')",
    }
    if "stages" in cfg:
        from engine.stage_scheduler import StageScheduler

        if epoch is None:
            raise ValueError("Staged training requires --train-epoch or a checkpoint containing epoch.")
        scheduler = StageScheduler(model, cfg["stages"])
        stage = scheduler.resolve_stage(epoch)
        for path in stage.freeze_modules:
            module = model.get_submodule(resolve_config_module_path(model, path))
            module.requires_grad_(False)
        if stage.detach_dsm_taps is not None:
            model.set_detach_dsm_taps(stage.detach_dsm_taps)
        metadata.update(
            freeze_policy="constructor+training_stage",
            stage_index=stage.index,
            stage_name=stage.name,
            freeze_modules=list(stage.freeze_modules),
            detach_dsm_taps=StageScheduler._get_detach_dsm_taps(model),
        )
    return RepositoryAdapter(model, metadata)
