from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from data import build_isprs_dataset
from engine.evaluator import Evaluator
from engine.mfnet_3090_trainer import MFNet3090Trainer
from engine.sliding_window_inferencer_3090 import SlidingWindowInferencer3090
from losses import build_loss
from models import build_model
from utils import (
    TestNetLogger,
    build_default_work_dir,
    build_optimizer_param_groups,
    load_config,
    log_run_summary,
    resolve_config_path,
    save_effective_config,
)
from utils.runtime_3090 import configure_3090_runtime


SUPPORTED_MODEL_TYPES = {
    "testnet_prealign_spmf20",
    "testnet_prealign_spmf21",
}


def collect_stage_lr_module_paths(stages: Any) -> list[str]:
    if not isinstance(stages, list):
        return []

    module_paths: list[str] = []
    for raw_stage in stages:
        if not isinstance(raw_stage, dict):
            continue
        module_lrs = raw_stage.get("module_lrs", {})
        if not isinstance(module_lrs, dict):
            continue
        for module_path in module_lrs:
            if isinstance(module_path, str) and module_path not in module_paths:
                module_paths.append(module_path)
    return module_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SPMF + Prealign on an Ampere-or-newer GPU using BF16."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config filename; searched directly under configs/ (for example, cfg_stage_d.jsonc).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument("--resume-ckpt", default=None)
    parser.add_argument("--load-from", default=None)
    return parser.parse_args()


def _load_resume_config(resume_dir: Path) -> dict[str, Any]:
    config_paths = sorted([*resume_dir.glob("*.jsonc"), *resume_dir.glob("*.json")])
    if not config_paths:
        raise FileNotFoundError(f"No JSON/JSONC config found in resume directory {resume_dir}.")
    return load_config(str(config_paths[0]))


def _validate_model_type(cfg: dict[str, Any]) -> str:
    model_type = str(cfg["model"]["type"])
    if model_type not in SUPPORTED_MODEL_TYPES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        raise ValueError(
            f"train_3090.py supports only {supported}, got {model_type!r}."
        )
    return model_type


def _build_loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    dataloader_cfg: dict[str, Any],
) -> DataLoader:
    num_workers = int(dataloader_cfg.get("num_workers", 0))
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(dataloader_cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            dataloader_cfg.get("persistent_workers", False)
        )
        loader_kwargs["prefetch_factor"] = int(
            dataloader_cfg.get("prefetch_factor", 2)
        )
    return DataLoader(dataset, **loader_kwargs)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if args.resume_dir is not None:
        work_dir = Path(args.resume_dir)
        cfg = _load_resume_config(work_dir)
        resume_from = str(work_dir / "latest.pth")
        load_from = None
    else:
        if args.config is None:
            raise ValueError("--config is required when --resume-dir is not provided.")
        config_path = resolve_config_path(args.config)
        cfg = load_config(config_path)
        model_type = _validate_model_type(cfg)
        resume_from = args.resume_ckpt
        load_from = args.load_from

        runtime_cfg = cfg.get("runtime_3090")
        if not isinstance(runtime_cfg, dict):
            raise KeyError("3090 config must define a runtime_3090 object.")
        dataset_cfg = cfg["dataset"]
        dataset_name = str(dataset_cfg.get("name", "vaihingen")).strip().lower()
        model_name = model_type
        work_dir = build_default_work_dir(
            model_name=model_name,
            dataset_name=dataset_name,
            lambda_align=cfg["train"].get("lambda_align"),
            root_dir=runtime_cfg.get("work_dir_root", "work_dirs"),
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        save_effective_config(cfg, work_dir / config_path.name)

    runtime_cfg = cfg.get("runtime_3090")
    if not isinstance(runtime_cfg, dict):
        raise KeyError("3090 config must define a runtime_3090 object.")
    model_type = _validate_model_type(cfg)
    configure_3090_runtime(
        device,
        enable_tf32=runtime_cfg.get("tf32", True),
    )

    dataset_cfg = cfg["dataset"]
    dataset_name = str(dataset_cfg.get("name", "vaihingen")).strip().lower()
    dataloader_cfg = dict(cfg.get("dataloader", {}))
    train_batch_size = int(cfg["train"]["batch_size"])
    val_batch_size = int(cfg["train"].get("val_batch_size", train_batch_size))

    model = build_model(cfg["model"])
    criterion = build_loss(
        cfg.get("loss"),
        weights=cfg.get("loss_weights"),
        class_weights=cfg.get("class_weights"),
    )

    train_dataset = build_isprs_dataset(
        dataset_name,
        root_dir=dataset_cfg["root_dir"],
        ids=dataset_cfg["train_ids"],
        patch_size=dataset_cfg.get("patch_size", [256, 256]),
        samples_per_epoch=dataset_cfg["train_samples_per_epoch"],
        cache=dataset_cfg.get("cache", True),
        augmentation=dataset_cfg.get("augmentation", True),
        dsm_preprocessing=dataset_cfg["dsm_preprocessing"],
        split="train",
        tile_sampling_weights=dataset_cfg.get("train_tile_sampling_weights"),
        patch_sampling=dataset_cfg.get("patch_sampling"),
    )
    train_loader = _build_loader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        dataloader_cfg=dataloader_cfg,
    )

    val_loader: Any = []
    if cfg["train"]["val_epoch_interval"] > 0:
        val_dataset = build_isprs_dataset(
            dataset_name,
            root_dir=dataset_cfg["root_dir"],
            ids=dataset_cfg["val_ids"],
            patch_size=dataset_cfg.get("patch_size", [256, 256]),
            samples_per_epoch=dataset_cfg.get("val_samples_per_epoch", len(dataset_cfg["val_ids"])),
            cache=dataset_cfg.get("cache", True),
            augmentation=False,
            dsm_preprocessing=dataset_cfg["dsm_preprocessing"],
            split="val",
        )
        val_loader = _build_loader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            dataloader_cfg=dataloader_cfg,
        )

    optimizer_cfg = cfg["optimizer"]
    base_lr = float(optimizer_cfg["lr"])
    weight_decay = float(optimizer_cfg.get("weight_decay", 5e-4))
    adapter_lr = float(optimizer_cfg.get("adapter_lr", base_lr))
    optimizer = torch.optim.SGD(
        build_optimizer_param_groups(
            model,
            weight_decay=weight_decay,
            base_lr=base_lr,
            adapter_lr=adapter_lr,
            lr_module_paths=collect_stage_lr_module_paths(cfg.get("stages")),
        ),
        lr=base_lr,
        momentum=optimizer_cfg.get("momentum", 0.9),
    )

    scheduler = None
    if cfg.get("scheduler") is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg["scheduler"].get("milestones", [25, 35, 45]),
            gamma=cfg["scheduler"].get("gamma", 0.1),
        )

    logger = TestNetLogger(
        str(work_dir),
        use_tensorboard=cfg["train"]["use_tensorboard"],
    )
    trainer_cfg = {
        **cfg["train"],
        "work_dir": str(work_dir),
        "experiment_name": work_dir.name or "mfnet",
        "resume_from": resume_from,
        "load_from": load_from,
        "sam_checkpoint": cfg["model"].get("sam_checkpoint"),
        "num_classes": cfg["model"]["num_classes"],
        "class_weights": cfg.get("class_weights"),
        "loss": cfg.get("loss"),
        "loss_weights": cfg.get("loss_weights"),
        "validation": cfg["validation"],
        "runtime_3090": runtime_cfg,
    }
    if cfg.get("stages") is not None:
        trainer_cfg["stages"] = cfg["stages"]

    trainer = MFNet3090Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        evaluator=Evaluator(),
        device=device,
        inferencer=SlidingWindowInferencer3090(),
        scheduler=scheduler,
        cfg=trainer_cfg,
    )

    if args.resume_dir is None:
        log_run_summary(
            logger=logger,
            model=model,
            work_dir=work_dir,
            experiment_name=work_dir.name or "mfnet",
        )
    trainer.train()


if __name__ == "__main__":
    main()
