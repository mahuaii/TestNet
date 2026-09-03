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
from engine import (
    Evaluator,
    MFNetAuxAlignDGATrainer,
    MFNetAuxAlignTrainer,
    MFNetBaselineAuxAlignTrainer,
    MFNetDGATrainer,
    MFNetTrainer,
    SlidingWindowInferencer,
)
from losses import build_loss
from models import build_model
from utils import (
    TestNetRecorderLogger,
    TestNetLogger,
    build_default_work_dir,
    build_optimizer_param_groups,
    load_config,
    log_run_summary,
    save_effective_config,
    resolve_config_path,
)

DGA_MODEL_TYPES = {
    "testnet_dga10",
    "testnet_dga20",
    "testnet_dga20_dgsf10",
    "testnet_dga10_softplus",
    "testnet_dga20_softplus",
    "testnet_dga30",
    "testnet_prealign_dga10",
    "testnet_prealign_auxalign_dga10",
}

RECORDER_LOGGER_MODEL_TYPES = DGA_MODEL_TYPES | {
    "testnet_dgsf10",
    "testnet_dgfm",
    "testnet_dgfm01",
    "testnet_dgfm01_upernet",
    "testnet_sgcf",
    "testnet_prealign_auxalign_dgsf10",
}

AUX_ALIGN_MODEL_TYPES = {
    "testnet_prealign_auxalign",
    "testnet_prealign_auxalign_dga10",
    "testnet_prealign_auxalign_dgsf10",
}

AUX_ALIGN_DGA_MODEL_TYPES = DGA_MODEL_TYPES & AUX_ALIGN_MODEL_TYPES
BASELINE_AUX_ALIGN_MODEL_TYPES = {
    "testnet_auxalign",
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Config filename; searched directly under configs/ (for example, cfg_stage_d.jsonc).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument("--resume-ckpt", default=None)
    parser.add_argument("--load-from", default=None)
    parser.add_argument("--model-type", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_type_override = getattr(args, "model_type", None)
    if args.resume_dir is not None:
        if model_type_override is not None:
            raise ValueError("--model-type cannot be used with --resume-dir; resume uses saved config")
        work_dir = Path(args.resume_dir)
        config_path = sorted([*work_dir.glob("*.jsonc"), *work_dir.glob("*.json")])[0]
        cfg = load_config(str(config_path))
        resume_from = str(work_dir / "latest.pth")
        load_from = None
    else:
        config_path = resolve_config_path(args.config)
        cfg = load_config(config_path)
        if model_type_override is not None:
            cfg["model"]["type"] = model_type_override
        dataset_cfg = cfg["dataset"]
        dataset_name = str(dataset_cfg.get("name", "vaihingen")).strip().lower()
        model_name = cfg["model"].get("type", "model")
        work_dir = build_default_work_dir(
            model_name=model_name,
            dataset_name=dataset_name,
            lambda_align=cfg["train"].get("lambda_align"),
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        save_effective_config(cfg, work_dir / config_path.name)
        resume_from = args.resume_ckpt
        load_from = args.load_from

    dataset_cfg = cfg["dataset"]
    dataset_name = str(dataset_cfg.get("name", "vaihingen")).strip().lower()
    experiment_name = work_dir.name or "mfnet"

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataloader"].get("num_workers", 0),
        pin_memory=True,
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
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["dataloader"].get("num_workers", 0),
            pin_memory=True,
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

    model_type = str(cfg["model"]["type"])
    if model_type in BASELINE_AUX_ALIGN_MODEL_TYPES:
        trainer_cls = MFNetBaselineAuxAlignTrainer
    elif model_type in AUX_ALIGN_DGA_MODEL_TYPES:
        trainer_cls = MFNetAuxAlignDGATrainer
    elif model_type in AUX_ALIGN_MODEL_TYPES:
        trainer_cls = MFNetAuxAlignTrainer
    elif model_type in DGA_MODEL_TYPES:
        trainer_cls = MFNetDGATrainer
    else:
        trainer_cls = MFNetTrainer
    logger_cls = TestNetRecorderLogger if model_type in RECORDER_LOGGER_MODEL_TYPES else TestNetLogger
    logger = logger_cls(str(work_dir), use_tensorboard=cfg["train"]["use_tensorboard"])

    trainer_cfg = {
        **cfg["train"],
        "work_dir": str(work_dir),
        "experiment_name": experiment_name,
        "resume_from": resume_from,
        "load_from": load_from,
        "sam_checkpoint": cfg["model"].get("sam_checkpoint"),
        "num_classes": cfg["model"]["num_classes"],
        "class_weights": cfg.get("class_weights"),
        "loss": cfg.get("loss"),
        "loss_weights": cfg.get("loss_weights"),
        "validation": cfg["validation"],
    }
    stages = cfg.get("stages")
    if stages is not None:
        trainer_cfg["stages"] = stages

    trainer = trainer_cls(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        evaluator=Evaluator(),
        device=torch.device(args.device),
        inferencer=SlidingWindowInferencer(),
        scheduler=scheduler,
        cfg=trainer_cfg,
    )
    # A --resume-dir run reuses the existing work_dir, so the initial run
    # summary in train.log should already be present and must not be duplicated.
    if args.resume_dir is None:
        log_run_summary(
            logger=logger,
            model=model,
            work_dir=work_dir,
            experiment_name=experiment_name,
        )
        if model_type in AUX_ALIGN_MODEL_TYPES or model_type in BASELINE_AUX_ALIGN_MODEL_TYPES:
            logger.log_message(f"Lambda align: {float(cfg['train'].get('lambda_align', 0.01)):.6f}")
            logger.log_message(f"Align index: {model.align_index}")
    trainer.train()


if __name__ == "__main__":
    main()
