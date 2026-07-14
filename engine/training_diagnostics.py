from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from utils import IntermediateStatsRecorder, LR_SCOPE_DEFAULT, resolve_config_module_path


DIAGNOSTIC_STAT_PREFIXES = ("prealign/", "spmf/")


def diagnostics_enabled(cfg: Mapping[str, Any], key: str | None = None) -> bool:
    diagnostics = cfg.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping) or not bool(diagnostics.get("enabled", False)):
        return False
    if key is None:
        return True
    return bool(diagnostics.get(key, False))


def attach_prealign_spmf_recorders(model: torch.nn.Module) -> None:
    stats = getattr(model, "intermediate_stats", None)
    if stats is None:
        stats = IntermediateStatsRecorder()
        model.intermediate_stats = stats

    _attach_module(model, "aux_prealign", stats, "prealign")
    _attach_module(model, "spmf_fusion", stats, "spmf")
    _attach_module(model, "structure_branch", stats, "spmf/structure")


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


def collect_trainable_param_counts(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, int]:
    return {
        "aux_prealign": _count_trainable_module(model, "aux_prealign"),
        "spmf_fusion": _count_trainable_module(model, "spmf_fusion"),
        "structure_branch": _count_trainable_module(model, "structure_branch"),
        "encoder_adapters": _count_encoder_params(model, adapter=True),
        "image_encoder_non_adapter": _count_encoder_params(model, adapter=False),
        "decoder": _count_trainable_module(model, "decoder"),
        "default_optimizer_scope": _count_trainable_default_scope(optimizer),
    }


def _attach_module(
    model: torch.nn.Module,
    module_path: str,
    stats: IntermediateStatsRecorder,
    prefix: str,
) -> None:
    try:
        module = model.get_submodule(resolve_config_module_path(model, module_path))
    except AttributeError:
        return
    module.intermediate_stats = stats
    module.intermediate_stats_prefix = prefix


def _count_params(params: object) -> int:
    return sum(param.numel() for param in params)  # type: ignore[union-attr]


def _count_trainable_module(model: torch.nn.Module, module_path: str) -> int:
    try:
        module = model.get_submodule(resolve_config_module_path(model, module_path))
    except AttributeError:
        return 0
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def _count_encoder_params(model: torch.nn.Module, *, adapter: bool) -> int:
    image_encoder = getattr(model, "image_encoder", None)
    if image_encoder is None:
        return 0
    return sum(
        param.numel()
        for name, param in image_encoder.named_parameters()
        if param.requires_grad and (("Adapter" in name) == adapter)
    )


def _count_trainable_default_scope(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for group in optimizer.param_groups:
        if str(group.get("lr_scope", LR_SCOPE_DEFAULT)) != LR_SCOPE_DEFAULT:
            continue
        total += sum(param.numel() for param in group["params"] if param.requires_grad)
    return total


def _display_lr_scope(lr_scope: str) -> str:
    if lr_scope == LR_SCOPE_DEFAULT:
        return "default"
    return lr_scope


__all__ = [
    "DIAGNOSTIC_STAT_PREFIXES",
    "attach_prealign_spmf_recorders",
    "collect_optimizer_group_summaries",
    "collect_trainable_param_counts",
    "diagnostics_enabled",
    "stage_label",
]
