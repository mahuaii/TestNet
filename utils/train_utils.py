from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .testnet_logger import TestNetLogger

GATE_WEIGHT_DECAY_EXEMPT_PARAM_NAMES = {"alpha", "beta", "gamma", "delta", "lambda"}
LR_SCOPE_DEFAULT = "__default__"
LR_SCOPE_DEFAULT_DISPLAY = "default"
OPTIMIZER_GROUP_METADATA_KEYS = (
    "group_name",
    "lr_scope",
    "is_adapter",
    "no_decay",
    "num_params",
    "nominal_lr",
)
LEGACY_MODULE_PATH_RENAMES = {
    "spmf10": "spmf_fusion10",
    "spmf11": "spmf_fusion11",
    "spmf20": "spmf_fusion20",
}
SPMF_CONFIG_MODULE_SPEC_ATTRS = {
    "structure_branch": "structure_attr",
    "spmf_fusion": "fusion_attr",
}
VERSIONED_SPMF_CONFIG_MODULE_RE = re.compile(r"^(structure_branch|spmf_fusion)\d+$")


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


def normalize_legacy_module_path(module_path: str) -> str:
    head, separator, tail = module_path.partition(".")
    renamed_head = LEGACY_MODULE_PATH_RENAMES.get(head)
    if renamed_head is None:
        return module_path
    if separator:
        return f"{renamed_head}.{tail}"
    return renamed_head


def normalize_legacy_state_dict_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    normalized_state: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = normalize_legacy_module_path(key)
        if normalized_key in normalized_state:
            raise ValueError(f"Duplicate model state key after legacy module path migration: {normalized_key}.")
        normalized_state[normalized_key] = value
    return normalized_state


def normalize_legacy_optimizer_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    normalized_state = dict(state_dict)
    param_groups = []
    for group in state_dict.get("param_groups", ()):
        normalized_group = dict(group)
        lr_scope = normalized_group.get("lr_scope")
        if isinstance(lr_scope, str):
            normalized_scope = _normalize_optimizer_lr_scope(lr_scope)
            normalized_group["lr_scope"] = normalized_scope
            group_name = normalized_group.get("group_name")
            if isinstance(group_name, str):
                group_scope, separator, group_suffix = group_name.partition(":")
                if separator:
                    normalized_group_scope = _normalize_optimizer_lr_scope(group_scope)
                    normalized_group["group_name"] = f"{normalized_group_scope}:{group_suffix}"
        param_groups.append(normalized_group)
    normalized_state["param_groups"] = param_groups
    return normalized_state


def _normalize_optimizer_lr_scope(lr_scope: str) -> str:
    normalized_scope = normalize_legacy_module_path(lr_scope)
    head, separator, tail = normalized_scope.partition(".")
    match = VERSIONED_SPMF_CONFIG_MODULE_RE.fullmatch(head)
    if match is None:
        return normalized_scope
    normalized_head = match.group(1)
    return f"{normalized_head}.{tail}" if separator else normalized_head


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
    for lr_scope in (LR_SCOPE_DEFAULT, *(config_path for config_path, _ in lr_scopes)):
        for is_adapter, no_decay in ((False, False), (False, True), (True, False), (True, True)):
            params = grouped_params.get((lr_scope, is_adapter, no_decay))
            if not params:
                continue
            scope_name = _display_lr_scope(lr_scope)
            adapter_name = "adapter" if is_adapter else "main"
            decay_name = "no_decay" if no_decay else "decay"
            group: dict[str, object] = {
                "params": params,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "group_name": f"{scope_name}:{adapter_name}:{decay_name}",
                "lr_scope": lr_scope,
                "is_adapter": is_adapter,
                "no_decay": no_decay,
                "num_params": sum(param.numel() for param in params),
            }
            group_lr = adapter_lr if is_adapter and adapter_lr is not None else base_lr
            if group_lr is not None:
                group["lr"] = group_lr
                group["nominal_lr"] = group_lr
            param_groups.append(group)
    return param_groups


def _validate_lr_module_paths(
    model: torch.nn.Module,
    lr_module_paths: Sequence[str] | None,
) -> tuple[tuple[str, str], ...]:
    if lr_module_paths is None:
        return ()
    if isinstance(lr_module_paths, str):
        raise TypeError("lr_module_paths must be a sequence of module path strings.")

    paths: list[tuple[str, str]] = []
    for module_path in lr_module_paths:
        if not isinstance(module_path, str) or not module_path:
            raise TypeError("lr_module_paths must contain non-empty strings.")
        validate_config_module_path(module_path)
        resolved_module_path = resolve_config_module_path(model, module_path)
        model.get_submodule(resolved_module_path)
        if any(config_path == module_path for config_path, _ in paths):
            raise ValueError(f"Duplicate lr module path: {module_path}.")
        paths.append((module_path, resolved_module_path))

    for index, (module_path, resolved_module_path) in enumerate(paths):
        for other_path, other_resolved_path in paths[index + 1:]:
            if resolved_module_path.startswith(f"{other_resolved_path}.") or other_resolved_path.startswith(
                f"{resolved_module_path}."
            ):
                raise ValueError(
                    "lr_module_paths must not contain overlapping module paths: "
                    f"{module_path!r}, {other_path!r}."
                )
    return tuple(paths)


def _resolve_lr_scope(name: str, lr_scopes: Sequence[tuple[str, str]]) -> str:
    for config_path, resolved_path in lr_scopes:
        if name.startswith(f"{resolved_path}."):
            return config_path
    return LR_SCOPE_DEFAULT


def validate_config_module_path(module_path: str) -> None:
    head = module_path.partition(".")[0]
    if head in LEGACY_MODULE_PATH_RENAMES or VERSIONED_SPMF_CONFIG_MODULE_RE.fullmatch(head):
        raise ValueError(
            f"Versioned SPMF config module path {module_path!r} is not supported; "
            "use 'structure_branch' or 'spmf_fusion'."
        )


def resolve_config_module_path(model: torch.nn.Module, module_path: str) -> str:
    validate_config_module_path(module_path)
    head, separator, tail = module_path.partition(".")
    spec_attr = SPMF_CONFIG_MODULE_SPEC_ATTRS.get(head)
    if spec_attr is None:
        return module_path

    spec = getattr(model, "_spmf_variant_spec", None)
    resolved_head = getattr(spec, spec_attr, None)
    if not isinstance(resolved_head, str) or not resolved_head:
        raise AttributeError(
            f"Model {type(model).__name__} does not define an SPMF module for config path {head!r}."
        )
    return f"{resolved_head}.{tail}" if separator else resolved_head


def _display_lr_scope(lr_scope: str) -> str:
    if lr_scope == LR_SCOPE_DEFAULT:
        return LR_SCOPE_DEFAULT_DISPLAY
    return lr_scope


def restore_optimizer_group_metadata(
    optimizer: torch.optim.Optimizer,
    metadata_by_group: Sequence[Mapping[str, object]],
) -> None:
    for group, metadata in zip(optimizer.param_groups, metadata_by_group):
        for key in OPTIMIZER_GROUP_METADATA_KEYS:
            if key not in group and key in metadata:
                group[key] = metadata[key]
        if "num_params" not in group:
            group["num_params"] = sum(param.numel() for param in group["params"])


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
