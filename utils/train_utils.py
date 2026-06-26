from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .testnet_logger import TestNetLogger

GATE_WEIGHT_DECAY_EXEMPT_PARAM_NAMES = {"alpha", "beta", "gamma", "delta", "lambda"}
LR_SCOPE_DEFAULT = "__default__"


def safe_path_component(value: object, fallback: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or fallback


def work_dir_model_suffix(model_name: object) -> str:
    model_component = safe_path_component(model_name, "model")
    tokens = [
        token
        for token in model_component.split("_")
        if token not in {"mfnet", "testnet", "unetformer"}
    ]
    return "_".join(tokens) or "base"


def build_default_work_dir(
    model_name: object,
    dataset_name: object,
    lambda_align: object | None = None,
    root_dir: str | Path = "work_dirs",
) -> Path:
    run_id = uuid.uuid4().hex[:5]
    name_parts = [
        safe_path_component(dataset_name, "dataset"),
        work_dir_model_suffix(model_name),
    ]
    if lambda_align is not None:
        name_parts.append(safe_path_component(f"lambda-{lambda_align}", "lambda"))
    name_parts.append(run_id)
    experiment_name = "_".join(name_parts)
    return Path(root_dir) / experiment_name

def count_model_params(model: torch.nn.Module) -> tuple[int, int, int, int]:
    all_params = sum(param.nelement() for param in model.parameters())
    image_encoder_params = 0
    adapter_params = 0
    for name, param in model.image_encoder.named_parameters():
        if "Adapter" in name:
            adapter_params += param.nelement()
        else:
            image_encoder_params += param.nelement()
    other_params = all_params - image_encoder_params - adapter_params
    return all_params, image_encoder_params, adapter_params, other_params


def is_gate_weight_decay_exempt_param(name: str) -> bool:
    return name.rsplit(".", maxsplit=1)[-1] in GATE_WEIGHT_DECAY_EXEMPT_PARAM_NAMES


def build_optimizer_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    *,
    base_lr: float | None = None,
    adapter_lr: float | None = None,
    lr_module_paths: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    if base_lr is not None and base_lr <= 0:
        raise ValueError(f"base_lr must be positive, got {base_lr}.")
    if adapter_lr is not None and adapter_lr <= 0:
        raise ValueError(f"adapter_lr must be positive, got {adapter_lr}.")

    lr_scopes = _validate_lr_module_paths(model, lr_module_paths)
    use_lr_scopes = bool(lr_scopes)
    grouped_params: dict[tuple[str, bool, bool], list[torch.nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr_scope = _resolve_lr_scope(name, lr_scopes)
        is_adapter = "Adapter" in name
        no_decay = param.ndim <= 1 or is_gate_weight_decay_exempt_param(name)
        key = (lr_scope, is_adapter, no_decay)
        grouped_params.setdefault(key, []).append(param)

    param_groups: list[dict[str, object]] = []
    for lr_scope in (LR_SCOPE_DEFAULT, *lr_scopes):
        for is_adapter, no_decay in ((False, False), (False, True), (True, False), (True, True)):
            params = grouped_params.get((lr_scope, is_adapter, no_decay))
            if not params:
                continue
            group: dict[str, object] = {
                "params": params,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
            group_lr = adapter_lr if is_adapter and adapter_lr is not None else base_lr
            if group_lr is not None:
                group["lr"] = group_lr
            if use_lr_scopes:
                group["lr_scope"] = lr_scope
                if group_lr is not None:
                    group["nominal_lr"] = group_lr
            param_groups.append(group)
    return param_groups


def _validate_lr_module_paths(
    model: torch.nn.Module,
    lr_module_paths: Sequence[str] | None,
) -> tuple[str, ...]:
    if lr_module_paths is None:
        return ()
    if isinstance(lr_module_paths, str):
        raise TypeError("lr_module_paths must be a sequence of module path strings.")

    paths: list[str] = []
    for module_path in lr_module_paths:
        if not isinstance(module_path, str) or not module_path:
            raise TypeError("lr_module_paths must contain non-empty strings.")
        model.get_submodule(module_path)
        if module_path in paths:
            raise ValueError(f"Duplicate lr module path: {module_path}.")
        paths.append(module_path)

    for index, module_path in enumerate(paths):
        for other_path in paths[index + 1:]:
            if module_path.startswith(f"{other_path}.") or other_path.startswith(f"{module_path}."):
                raise ValueError(
                    "lr_module_paths must not contain overlapping module paths: "
                    f"{module_path!r}, {other_path!r}."
                )
    return tuple(paths)


def _resolve_lr_scope(name: str, lr_scopes: Sequence[str]) -> str:
    for lr_scope in lr_scopes:
        if name.startswith(f"{lr_scope}."):
            return lr_scope
    return LR_SCOPE_DEFAULT


def log_run_summary(
    logger: TestNetLogger,
    model: torch.nn.Module,
    work_dir: Path,
    experiment_name: str,
) -> None:
    all_params, image_encoder_params, adapter_params, other_params = count_model_params(model)
    logger.log_message(f"Experiment: {experiment_name}")
    logger.log_message(f"Workdir: {work_dir}")
    logger.log_message(f"All Params:   {all_params}")
    logger.log_message(f"ImgEncoder:   {image_encoder_params}")
    logger.log_message(f"Adapter: {adapter_params}")
    logger.log_message(f"Others: {other_params}")


def save_effective_config(cfg: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(cfg, indent=4) + "\n", encoding="utf-8")
