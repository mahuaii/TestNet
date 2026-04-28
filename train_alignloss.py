from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from data import build_isprs_dataset
from engine.evaluator import Evaluator
from engine.mfnet_auxalign_trainer import MFNetAuxAlignTrainer
from engine.sliding_window_inferencer import SlidingWindowInferencer
from models import build_model
from train import build_default_work_dir, log_run_summary, set_reproducibility
from utils import MFNetLogger, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config_prealign_alignloss.jsonc")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument("--resume-ckpt", default=None)
    parser.add_argument("--load-from", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume_dir is not None:
        work_dir = Path(args.resume_dir)
        config_path = sorted([*work_dir.glob("*.jsonc"), *work_dir.glob("*.json")])[0]
        cfg = load_config(str(config_path))
        resume_from = str(work_dir / "latest.pth")
        load_from = None
    else:
        cfg = load_config(args.config)
        dataset_cfg = cfg["dataset"]
        dataset_name = str(dataset_cfg.get("name", "vaihingen")).strip().lower()
        model_name = cfg["model"].get("type", "model")
        work_dir = build_default_work_dir(
            model_name=model_name,
            dataset_name=dataset_name,
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, work_dir)
        resume_from = args.resume_ckpt
        load_from = args.load_from

    dataset_cfg = cfg["dataset"]
    dataset_name = str(dataset_cfg.get("name", "vaihingen")).strip().lower()
    experiment_name = work_dir.name or "mfnet_prealign_alignloss"
    seed = int(cfg.get("seed", 80))
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

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        momentum=cfg["optimizer"].get("momentum", 0.9),
        weight_decay=cfg["optimizer"].get("weight_decay", 5e-4),
    )

    scheduler = None
    if cfg.get("scheduler") is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg["scheduler"].get("milestones", [25, 35, 45]),
            gamma=cfg["scheduler"].get("gamma", 0.1),
        )

    logger = MFNetLogger(str(work_dir), use_tensorboard=cfg["train"]["use_tensorboard"])
    trainer = MFNetAuxAlignTrainer(
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
    if args.resume_dir is None:
        log_run_summary(
            logger=logger,
            model=model,
            work_dir=work_dir,
            experiment_name=experiment_name,
            seed=seed,
        )
        logger.log_message(f"Lambda align: {float(cfg['train'].get('lambda_align', 0.01)):.6f}")
        logger.log_message(f"Align index: {model.align_index}")
    trainer.train()


if __name__ == "__main__":
    main()
