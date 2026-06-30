from __future__ import annotations

from typing import Any

import torch

from utils import LR_SCOPE_DEFAULT


def collect_optimizer_group_summaries(
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for index, group in enumerate(optimizer.param_groups):
        lr_scope = str(group.get("lr_scope", LR_SCOPE_DEFAULT))
        display_scope = _display_lr_scope(lr_scope)
        group_name = str(group.get("group_name", f"{display_scope}:group_{index}"))
        effective_lr = float(group["lr"])
        nominal_lr = float(group.get("nominal_lr", effective_lr))
        summaries.append(
            {
                "index": index,
                "group_name": group_name,
                "lr_scope": display_scope,
                "nominal_lr": nominal_lr,
                "effective_lr": effective_lr,
                "num_params": int(group.get("num_params", _count_params(group["params"]))),
            }
        )
    return summaries


def stage_label(stage: Any | None) -> str:
    if stage is None:
        return "none"
    name = getattr(stage, "name", None)
    if name:
        return str(name)
    index = int(getattr(stage, "index", 0)) + 1
    return f"stage_{index}"


def _count_params(params: object) -> int:
    return sum(param.numel() for param in params)  # type: ignore[union-attr]


def _display_lr_scope(lr_scope: str) -> str:
    if lr_scope == LR_SCOPE_DEFAULT:
        return "default"
    return lr_scope


__all__ = [
    "collect_optimizer_group_summaries",
    "stage_label",
]
