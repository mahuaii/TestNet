from __future__ import annotations

import json
import random
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .mfnet_logger import MFNetLogger

GATE_WEIGHT_DECAY_EXEMPT_PARAM_NAMES = {"alpha", "beta", "gamma", "lambda"}


def safe_path_component(value: object, fallback: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or fallback


def work_dir_model_suffix(model_name: object) -> str:
    model_component = safe_path_component(model_name, "model")
    tokens = [token for token in model_component.split("_") if token not in {"mfnet", "unetformer"}]
    return "_".join(tokens) or "base"


def build_default_work_dir(
    model_name: object,
    dataset_name: object,
    seed: object,
    lambda_align: object | None = None,
    root_dir: str | Path = "work_dirs",
) -> Path:
    run_id = uuid.uuid4().hex[:5]
    name_parts = [
        safe_path_component(dataset_name, "dataset"),
        work_dir_model_suffix(model_name),
        safe_path_component(seed, "seed"),
        run_id,
    ]
    if lambda_align is not None:
        name_parts.append(safe_path_component(f"lambda-{lambda_align}", "lambda"))
    experiment_name = "_".join(name_parts)
    return Path(root_dir) / experiment_name


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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


def build_optimizer_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, object]]:
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if is_gate_weight_decay_exempt_param(name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups: list[dict[str, object]] = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return param_groups


def log_run_summary(
    logger: MFNetLogger,
    model: torch.nn.Module,
    work_dir: Path,
    experiment_name: str,
    seed: int,
) -> None:
    all_params, image_encoder_params, adapter_params, other_params = count_model_params(model)
    logger.log_message(f"Experiment: {experiment_name}")
    logger.log_message(f"Workdir: {work_dir}")
    logger.log_message(f"Seed: {seed}")
    logger.log_message(f"All Params:   {all_params}")
    logger.log_message(f"ImgEncoder:   {image_encoder_params}")
    logger.log_message(f"Adapter: {adapter_params}")
    logger.log_message(f"Others: {other_params}")


def save_effective_config(cfg: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(cfg, indent=4) + "\n", encoding="utf-8")
