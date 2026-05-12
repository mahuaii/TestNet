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
    MFNetAuxAlignTrainer,
    MFNetDGAContributionStatsTrainer,
    MFNetDGATrainer,
    MFNetTrainer,
    SlidingWindowInferencer,
)
from models import build_model
from utils import (
    MFNetDGALogger,
    MFNetLogger,
    build_default_work_dir,
    build_optimizer_param_groups,
    load_config,
    log_run_summary,
    save_effective_config,
    set_reproducibility,
)

DGA_MODEL_TYPES = {
    "mfnet_unetformer_dga10",
    "mfnet_unetformer_dga20",
    "mfnet_unetformer_dga10_contrib_stats",
    "mfnet_unetformer_dga20_contrib_stats",
    "mfnet_unetformer_dga10_softplus",
    "mfnet_unetformer_dga20_softplus",
    "mfnet_unetformer_dga30",
    "mfnet_unetformer_prealign_dga10",
}

DGA_CONTRIB_STATS_MODEL_TYPES = {
    "mfnet_unetformer_dga10_contrib_stats",
    "mfnet_unetformer_dga20_contrib_stats",
    "mfnet_unetformer_dga10_softplus",
    "mfnet_unetformer_dga20_softplus",
}

AUX_ALIGN_MODEL_TYPES = {
    "mfnet_unetformer_prealign_auxalign",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument("--resume-ckpt", default=None)
    parser.add_argument("--load-from", default=None)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_type_override = getattr(args, "model_type", None)
    seed_override = getattr(args, "seed", None)
    if args.resume_dir is not None:
        if model_type_override is not None:
            raise ValueError("--model-type cannot be used with --resume-dir; resume uses saved config")
        if seed_override is not None:
            raise ValueError("--seed cannot be used with --resume-dir; resume uses saved config")
        work_dir = Path(args.resume_dir)
        config_path = sorted([*work_dir.glob("*.jsonc"), *work_dir.glob("*.json")])[0]
        cfg = load_config(str(config_path))
        resume_from = str(work_dir / "latest.pth")
        load_from = None
    else:
        cfg = load_config(args.config)
        if model_type_override is not None:
            cfg["model"]["type"] = model_type_override
        if seed_override is not None:
            cfg["seed"] = seed_override
        dataset_cfg = cfg["dataset"]
        dataset_name = str(dataset_cfg.get("name", "vaihingen")).strip().lower()
        model_name = cfg["model"].get("type", "model")
        work_dir = build_default_work_dir(
            model_name=model_name,
            dataset_name=dataset_name,
            seed=cfg["seed"],
            lambda_align=cfg["train"].get("lambda_align"),
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        save_effective_config(cfg, work_dir / Path(args.config).name)
        resume_from = args.resume_ckpt
        load_from = args.load_from

    dataset_cfg = cfg["dataset"]
    dataset_name = str(dataset_cfg.get("name", "vaihingen")).strip().lower()
    experiment_name = work_dir.name or "mfnet"
    seed = int(cfg["seed"])
    set_reproducibility(seed)

    model = build_model(cfg["model"])

    train_dataset = build_isprs_dataset(
        dataset_name,
        root_dir=dataset_cfg["root_dir"],
        ids=dataset_cfg["train_ids"],
        patch_size=dataset_cfg.get("patch_size", [256, 256]),
        samples_per_epoch=dataset_cfg["train_samples_per_epoch"],
        cache=dataset_cfg.get("cache", True),
        augmentation=dataset_cfg.get("augmentation", True),
        split="train",
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataloader"].get("num_workers", 0),
        pin_memory=True,
        generator=train_generator,
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
            split="val",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["dataloader"].get("num_workers", 0),
            pin_memory=True,
        )

    weight_decay = cfg["optimizer"].get("weight_decay", 5e-4)
    optimizer = torch.optim.SGD(
        build_optimizer_param_groups(model, weight_decay=weight_decay),
        lr=cfg["optimizer"]["lr"],
        momentum=cfg["optimizer"].get("momentum", 0.9),
    )

    scheduler = None
    if cfg.get("scheduler") is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg["scheduler"].get("milestones", [25, 35, 45]),
            gamma=cfg["scheduler"].get("gamma", 0.1),
        )

    model_type = str(cfg["model"]["type"])
    if model_type in AUX_ALIGN_MODEL_TYPES:
        trainer_cls = MFNetAuxAlignTrainer
    elif model_type in DGA_CONTRIB_STATS_MODEL_TYPES:
        trainer_cls = MFNetDGAContributionStatsTrainer
    elif model_type in DGA_MODEL_TYPES:
        trainer_cls = MFNetDGATrainer
    else:
        trainer_cls = MFNetTrainer
    logger_cls = MFNetDGALogger if model_type in DGA_MODEL_TYPES else MFNetLogger
    logger = logger_cls(str(work_dir), use_tensorboard=cfg["train"]["use_tensorboard"])

    trainer = trainer_cls(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        evaluator=Evaluator(),
        device=torch.device(args.device),
        inferencer=SlidingWindowInferencer(),
        scheduler=scheduler,
        cfg={
            **cfg["train"],
            "work_dir": str(work_dir),
            "experiment_name": experiment_name,
            "resume_from": resume_from,
            "load_from": load_from,
            "seed": seed,
            "log_seed_after_resume": args.resume_dir is not None,
            "sam_checkpoint": cfg["model"].get("sam_checkpoint"),
            "num_classes": cfg["model"]["num_classes"],
            "class_weights": cfg.get("class_weights"),
            "validation": cfg["validation"],
        },
    )
    # A --resume-dir run reuses the existing work_dir and train.log, so the
    # initial run summary should already be present and must not be duplicated.
    if args.resume_dir is None:
        log_run_summary(
            logger=logger,
            model=model,
            work_dir=work_dir,
            experiment_name=experiment_name,
            seed=seed,
        )
        if model_type in AUX_ALIGN_MODEL_TYPES:
            logger.log_message(f"Lambda align: {float(cfg['train'].get('lambda_align', 0.01)):.6f}")
            logger.log_message(f"Align index: {model.align_index}")
    trainer.train()


if __name__ == "__main__":
    main()
