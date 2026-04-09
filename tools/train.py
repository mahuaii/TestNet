from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from datasets import VAIHINGEN_TRAIN_IDS, VAIHINGEN_VAL_IDS, VaihingenDataset
from engine import Evaluator, Inferencer, MFNetTrainer
from models import build_model
from utils import CheckpointManager, MFNetLogger, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.json")
    parser.add_argument("--work-dir", default="work_dirs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--load-from", default=None)
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_type = str(cfg["type"])
    if optimizer_type != "SGD":
        raise KeyError(f"Unsupported optimizer type: {optimizer_type!r}. Only 'SGD' is supported.")
    return torch.optim.SGD(
        model.parameters(),
        lr=float(cfg["lr"]),
        momentum=float(cfg.get("momentum", 0.9)),
        weight_decay=float(cfg.get("weight_decay", 5e-4)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict[str, Any] | None
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if cfg is None:
        return None
    scheduler_type = str(cfg["type"])
    if scheduler_type != "MultiStepLR":
        raise KeyError(
            f"Unsupported scheduler type: {scheduler_type!r}. Only 'MultiStepLR' is supported."
        )
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(item) for item in cfg.get("milestones", [25, 35, 45])],
        gamma=float(cfg.get("gamma", 0.1)),
    )


def build_train_dataset(cfg: dict[str, Any]) -> VaihingenDataset:
    return VaihingenDataset(
        root_dir=cfg["root_dir"],
        ids=cfg.get("train_ids", VAIHINGEN_TRAIN_IDS),
        patch_size=cfg.get("patch_size", [256, 256]),
        samples_per_epoch=int(cfg.get("train_samples_per_epoch", cfg.get("samples_per_epoch", 1000))),
        cache=bool(cfg.get("cache", True)),
        augmentation=bool(cfg.get("augmentation", True)),
        split="train",
    )


def build_val_dataset(cfg: dict[str, Any]) -> VaihingenDataset:
    return VaihingenDataset(
        root_dir=cfg["root_dir"],
        ids=cfg.get("val_ids", VAIHINGEN_VAL_IDS),
        patch_size=cfg.get("patch_size", [256, 256]),
        samples_per_epoch=int(cfg.get("val_samples_per_epoch", len(cfg.get("val_ids", VAIHINGEN_VAL_IDS)))),
        cache=bool(cfg.get("cache", True)),
        augmentation=False,
        split="val",
    )


def resolve_resume_from(
    explicit_resume_from: str | None,
    train_cfg: dict[str, Any],
    work_dir: Path,
) -> str | None:
    if explicit_resume_from:
        return explicit_resume_from
    if not bool(train_cfg.get("auto_resume", False)):
        return None

    latest_checkpoint = work_dir / "latest.pth"
    if latest_checkpoint.is_file():
        return str(latest_checkpoint)
    return None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]
    dataloader_cfg = cfg["dataloader"]
    optimizer_cfg = cfg["optimizer"]
    scheduler_cfg = cfg.get("scheduler")
    train_cfg = dict(cfg["train"])
    train_cfg.setdefault("batch_size", int(dataloader_cfg.get("batch_size", 1)))
    train_cfg.setdefault("effective_batch_size", int(train_cfg["batch_size"]))
    train_cfg.setdefault("val_epoch_interval", 0)
    train_cfg.setdefault("log_step_interval", 1)
    train_cfg.setdefault("save_epoch_interval", 1)
    train_cfg.setdefault("save_step_interval", 0)
    train_cfg.setdefault("use_tensorboard", True)
    train_cfg.setdefault("auto_resume", False)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = work_dir.name or "mfnet"
    resume_from = resolve_resume_from(args.resume_from, train_cfg, work_dir)

    model = build_model(model_cfg)
    train_dataset = build_train_dataset(dataset_cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(dataloader_cfg.get("num_workers", 0)),
    )

    val_loader: Any = []
    if int(train_cfg["val_epoch_interval"]) > 0:
        val_dataset = build_val_dataset(dataset_cfg)
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(dataloader_cfg.get("num_workers", 0)),
        )

    optimizer = build_optimizer(model, optimizer_cfg)
    scheduler = build_scheduler(optimizer, scheduler_cfg)

    trainer = MFNetTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=MFNetLogger(str(work_dir), use_tensorboard=bool(train_cfg["use_tensorboard"])),
        checkpoint_manager=CheckpointManager(str(work_dir)),
        evaluator=Evaluator(),
        device=torch.device(args.device),
        inferencer=Inferencer(),
        scheduler=scheduler,
        cfg={
            **train_cfg,
            "work_dir": str(work_dir),
            "experiment_name": experiment_name,
            "resume_from": resume_from,
            "load_from": args.load_from,
            "sam_checkpoint": model_cfg.get("sam_checkpoint"),
            "num_classes": int(model_cfg["num_classes"]),
            "class_weights": cfg.get("class_weights"),
        },
    )
    trainer.train()


if __name__ == "__main__":
    main()
