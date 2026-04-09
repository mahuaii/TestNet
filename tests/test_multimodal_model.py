from __future__ import annotations

import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from datasets import VaihingenDataset
from engine import MFNetTrainer
from utils import CheckpointManager, MFNetLogger


class _SingleBatchDataset(Dataset):
    def __init__(self, batch: dict[str, object]) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, object]:
        del index
        return self.batch


class _CaptureModel(torch.nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.last_call: tuple[torch.Tensor, torch.Tensor, str] | None = None

    def forward(self, rgb: torch.Tensor, dsm: torch.Tensor, mode: str = "Train") -> torch.Tensor:
        self.last_call = (rgb.detach().clone(), dsm.detach().clone(), str(mode))
        logits = torch.stack(
            [
                rgb[:, 0] - self.weight,
                rgb[:, 0] + self.weight,
            ],
            dim=1,
        )
        return logits


class MFNetTrainingTest(unittest.TestCase):
    def _build_trainer(
        self,
        tmpdir: str,
        model: torch.nn.Module,
        sample: dict[str, object],
    ) -> MFNetTrainer:
        train_loader = DataLoader(_SingleBatchDataset(sample), batch_size=1, shuffle=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)
        return MFNetTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=[],
            logger=MFNetLogger(tmpdir),
            checkpoint_manager=CheckpointManager(tmpdir),
            evaluator=None,
            inferencer=None,
            device=torch.device("cpu"),
            cfg={
                "max_epochs": 1,
                "batch_size": 1,
                "effective_batch_size": 1,
                "log_step_interval": 1,
                "val_epoch_interval": 0,
                "save_epoch_interval": 1,
                "save_step_interval": 0,
            },
        )

    def test_train_forward_calls_mfnet_signature_without_modifying_dsm(self) -> None:
        sample = {
            "inputs": {
                "rgb": torch.rand(3, 16, 16),
                "dsm": torch.rand(16, 16),
            },
            "target": torch.randint(0, 2, (16, 16), dtype=torch.long),
            "meta": {"tile_id": "1"},
        }
        model = _CaptureModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._build_trainer(tmpdir, model, sample)
            batch = next(iter(trainer.train_loader))

            loss, metrics = trainer.train_forward(batch)

            self.assertEqual(loss.ndim, 0)
            self.assertIn("loss", metrics)
            self.assertIn("accuracy", metrics)
            assert model.last_call is not None
            _, dsm, mode = model.last_call
            self.assertEqual(dsm.shape, (1, 16, 16))
            self.assertEqual(mode, "Train")

    def test_train_forward_rejects_unexpected_dsm_shape(self) -> None:
        sample = {
            "inputs": {
                "rgb": torch.rand(3, 12, 12),
                "dsm": torch.rand(1, 12, 12),
            },
            "target": torch.randint(0, 2, (12, 12), dtype=torch.long),
            "meta": {"tile_id": "1"},
        }
        model = _CaptureModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._build_trainer(tmpdir, model, sample)
            batch = next(iter(trainer.train_loader))

            with self.assertRaisesRegex(ValueError, "expected DSM with shape \\[B, H, W\\]"):
                trainer.train_forward(batch)

    def test_train_forward_rejects_non_long_target(self) -> None:
        sample = {
            "inputs": {
                "rgb": torch.rand(3, 12, 12),
                "dsm": torch.rand(12, 12),
            },
            "target": torch.randint(0, 2, (12, 12), dtype=torch.int32),
            "meta": {"tile_id": "1"},
        }
        model = _CaptureModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._build_trainer(tmpdir, model, sample)
            batch = next(iter(trainer.train_loader))

            with self.assertRaisesRegex(TypeError, "expected target dtype torch.long"):
                trainer.train_forward(batch)

    def test_vaihingen_dataset_emits_real_training_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "rgb").mkdir()
            (root / "dsm").mkdir()
            (root / "labels").mkdir()

            rgb = torch.zeros(32, 32, 3, dtype=torch.uint8).numpy()
            rgb[:, :, 0] = 255
            dsm = (torch.arange(32 * 32, dtype=torch.int16).reshape(32, 32)).numpy()
            label = torch.zeros(32, 32, 3, dtype=torch.uint8).numpy()
            label[:, :16] = torch.tensor([255, 255, 255], dtype=torch.uint8).numpy()
            label[:, 16:] = torch.tensor([0, 0, 255], dtype=torch.uint8).numpy()

            Image.fromarray(rgb).save(root / "rgb" / "top_mosaic_09cm_area1.tif")
            Image.fromarray(dsm).save(root / "dsm" / "dsm_09cm_matching_area1.tif")
            Image.fromarray(label).save(root / "labels" / "top_mosaic_09cm_area1.tif")

            random.seed(0)
            dataset = VaihingenDataset(
                root_dir=str(root),
                ids=["1"],
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=False,
                split="val",
            )

            sample = dataset[0]

            self.assertIn("inputs", sample)
            self.assertEqual(sample["inputs"]["rgb"].shape, (3, 16, 16))
            self.assertEqual(sample["inputs"]["dsm"].shape, (16, 16))
            self.assertEqual(sample["target"].shape, (16, 16))
            self.assertTrue(set(torch.unique(sample["target"]).tolist()).issubset({0, 1}))

    def test_train_entry_builds_mfnet_trainer_with_sgd_and_scheduler(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "test_train_module",
            Path(__file__).resolve().parents[1] / "tools" / "train.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

        captured_dataset_calls: list[dict[str, object]] = []
        captured_trainer_kwargs: list[dict[str, object]] = []

        class FakeTrainer:
            def __init__(self, **kwargs: object) -> None:
                captured_trainer_kwargs.append(kwargs)

            def train(self) -> None:
                return None

        original_parse_args = module.parse_args
        original_load_config = module.load_config
        original_build_model = module.build_model
        original_dataset = module.VaihingenDataset
        original_dataloader = module.DataLoader
        original_trainer = module.MFNetTrainer
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                module.parse_args = lambda: type(
                    "Args",
                    (),
                    {
                        "config": "unused.json",
                        "work_dir": tmpdir,
                        "device": "cpu",
                        "resume_from": None,
                        "load_from": None,
                    },
                )()
                module.load_config = lambda _path: {
                    "model": {"type": "mfnet_unetformer", "num_classes": 6},
                    "dataset": {
                        "root_dir": tmpdir,
                        "patch_size": [32, 32],
                        "train_ids": ["1"],
                        "val_ids": ["5"],
                        "train_samples_per_epoch": 4,
                        "val_samples_per_epoch": 1,
                        "cache": True,
                        "augmentation": True,
                    },
                    "dataloader": {"num_workers": 0},
                    "optimizer": {
                        "type": "SGD",
                        "lr": 0.01,
                        "momentum": 0.9,
                        "weight_decay": 0.0005
                    },
                    "scheduler": {
                        "type": "MultiStepLR",
                        "milestones": [1, 2],
                        "gamma": 0.1
                    },
                    "train": {
                        "max_epochs": 1,
                        "batch_size": 2,
                        "effective_batch_size": 2,
                        "log_step_interval": 1,
                        "val_epoch_interval": 0,
                        "save_epoch_interval": 1,
                        "save_step_interval": 0
                    },
                }
                module.build_model = lambda _cfg: FakeModel()

                def fake_dataset(**kwargs: object) -> dict[str, object]:
                    captured_dataset_calls.append(kwargs)
                    return {"dataset_kwargs": kwargs}

                module.VaihingenDataset = fake_dataset
                module.DataLoader = lambda dataset, **kwargs: {
                    "dataset": dataset,
                    "batch_size": kwargs["batch_size"],
                    "loader_kwargs": kwargs,
                }
                module.MFNetTrainer = FakeTrainer

                module.main()

                self.assertEqual(len(captured_dataset_calls), 1)
                self.assertEqual(captured_dataset_calls[0]["split"], "train")
                self.assertEqual(len(captured_trainer_kwargs), 1)
                trainer_kwargs = captured_trainer_kwargs[0]
                self.assertIsInstance(trainer_kwargs["optimizer"], torch.optim.SGD)
                self.assertIsInstance(
                    trainer_kwargs["scheduler"],
                    torch.optim.lr_scheduler.MultiStepLR,
                )
                self.assertIsInstance(trainer_kwargs["logger"], MFNetLogger)
                self.assertTrue(trainer_kwargs["logger"].use_tensorboard)
                self.assertEqual(trainer_kwargs["cfg"]["val_epoch_interval"], 0)
                self.assertEqual(trainer_kwargs["cfg"]["batch_size"], 2)
        finally:
            module.parse_args = original_parse_args
            module.load_config = original_load_config
            module.build_model = original_build_model
            module.VaihingenDataset = original_dataset
            module.DataLoader = original_dataloader
            module.MFNetTrainer = original_trainer


if __name__ == "__main__":
    unittest.main()
