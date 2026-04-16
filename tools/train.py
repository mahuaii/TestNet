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

from datasets import get_default_isprs_tile_ids, build_isprs_dataset
from engine import Evaluator, MFNetTrainer, SlidingWindowInferencer
from models import build_model
from utils import MFNetLogger, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.json")
    # TODO：默认实验路径为./workdirs，实验文件夹名称使用时间戳_模型名_数据集
    parser.add_argument("--work-dir", default="work_dirs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--load-from", default=None)
    return parser.parse_args()


def count_model_params(model: torch.nn.Module) -> tuple[int, int, int, int]:
    all_params = sum(param.nelement() for param in model.parameters())
    image_encoder_params = 0
    lora_params = 0
    for name, param in model.image_encoder.named_parameters():
        if "lora_" in name:
            lora_params += param.nelement()
        else:
            image_encoder_params += param.nelement()
    other_params = all_params - image_encoder_params - lora_params
    return all_params, image_encoder_params, lora_params, other_params


def log_run_summary(
    logger: MFNetLogger,
    model: torch.nn.Module,
    work_dir: Path,
    experiment_name: str,
) -> None:
    all_params, image_encoder_params, lora_params, other_params = count_model_params(model)
    logger.log_message(f"Experiment: {experiment_name}")
    logger.log_message(f"Workdir: {work_dir}")
    logger.log_message(f"All Params:   {all_params}")
    logger.log_message(f"ImgEncoder:   {image_encoder_params}")
    logger.log_message(f"Lora: {lora_params}")
    logger.log_message(f"Others: {other_params}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = work_dir.name or "mfnet"
    resume_from = args.resume_from
    if resume_from is None and cfg["train"].get("auto_resume", False):
        resume_from = str(work_dir / "latest.pth")

    model = build_model(cfg["model"])

    dataset_cfg = cfg["dataset"]
    dataset_name = str(dataset_cfg.get("name", "vaihingen")).strip().lower()
    default_train_ids, default_val_ids = get_default_isprs_tile_ids(dataset_name)

    train_dataset = build_isprs_dataset(
        dataset_name,
        root_dir=dataset_cfg["root_dir"],
        ids=dataset_cfg.get("train_ids", default_train_ids),
        patch_size=dataset_cfg.get("patch_size", [256, 256]),
        samples_per_epoch=dataset_cfg["train_samples_per_epoch"],
        cache=dataset_cfg.get("cache", True),
        augmentation=dataset_cfg.get("augmentation", True),
        split="train",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataloader"].get("num_workers", 0),
    )

    val_loader: Any = []
    if cfg["train"]["val_epoch_interval"] > 0:
        default_val_ids = dataset_cfg.get("val_ids", default_val_ids)
        val_dataset = build_isprs_dataset(
            dataset_name,
            root_dir=dataset_cfg["root_dir"],
            ids=default_val_ids,
            patch_size=dataset_cfg.get("patch_size", [256, 256]),
            samples_per_epoch=dataset_cfg.get("val_samples_per_epoch", len(default_val_ids)),
            cache=dataset_cfg.get("cache", True),
            augmentation=False,
            split="val",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["dataloader"].get("num_workers", 0),
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

    trainer = MFNetTrainer(
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
            "load_from": args.load_from,
            "sam_checkpoint": cfg["model"].get("sam_checkpoint"),
            "num_classes": cfg["model"]["num_classes"],
            "class_weights": cfg.get("class_weights"),
            "validation": cfg["validation"],
        },
    )
    log_run_summary(
        logger=logger,
        model=model,
        work_dir=work_dir,
        experiment_name=experiment_name,
    )
    trainer.train()


if __name__ == "__main__":
    main()
