from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from datasets import DummySegDataset
from engine import Evaluator, Trainer
from models import build_model
from utils import CheckpointManager, Logger, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--work-dir", default="work_dirs/minimal")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--load-from", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    work_dir = args.work_dir
    dataset_cfg = cfg["dataset"]
    dataloader_cfg = cfg["dataloader"]
    optimizer_cfg = cfg["optimizer"]

    Path(work_dir).mkdir(parents=True, exist_ok=True)

    model = build_model(cfg["model"])
    num_classes = int(model.num_classes)
    rgb_key = model.rgb_key

    train_dataset = DummySegDataset(
        length=dataset_cfg["train_samples"],
        image_size=dataset_cfg["image_size"],
        num_classes=num_classes,
        rgb_key=rgb_key,
    )
    val_dataset = DummySegDataset(
        length=dataset_cfg["val_samples"],
        image_size=dataset_cfg["image_size"],
        num_classes=num_classes,
        rgb_key=rgb_key,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader_cfg["batch_size"],
        shuffle=True,
        num_workers=dataloader_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataloader_cfg["batch_size"],
        shuffle=False,
        num_workers=dataloader_cfg["num_workers"],
    )

    optimizer_cls = getattr(torch.optim, optimizer_cfg["type"])
    optimizer = optimizer_cls(model.parameters(), lr=optimizer_cfg["lr"])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=Logger(work_dir),
        checkpoint_manager=CheckpointManager(work_dir),
        evaluator=Evaluator(),
        device=torch.device(args.device),
        cfg={
            **train_cfg,
            "work_dir": work_dir,
            "resume_from": args.resume_from,
            "load_from": args.load_from,
            "num_classes": num_classes,
        },
    )
    trainer.train()


if __name__ == "__main__":
    main()
