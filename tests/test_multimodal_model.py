from __future__ import annotations

import importlib
import importlib.util
import random
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import call, patch

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from data.isprs_dataset import (
    ISPRSDataset,
    POTSDAM_PRESET,
    VAIHINGEN_PRESET,
)
from data import build_isprs_dataset
from engine import MFNetTrainer, SlidingWindowInferencer
from utils import DataUtils, MFNetLogger


class _SingleBatchDataset(Dataset):
    def __init__(self, batch: dict[str, object]) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, object]:
        del index
        return self.batch


class _ListDataset(Dataset):
    def __init__(self, batches: list[dict[str, object]]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.batches[index]


class _DelegatingMFNetTrainer(MFNetTrainer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.compute_loss_and_metrics_called = False

    def _compute_loss_and_metrics(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del logits, target
        self.compute_loss_and_metrics_called = True
        loss = self.model.weight.sum() * 0.0 + 3.0
        return loss, {"loss": 3.0, "accuracy": 25.0}


class _WholeTileDataset(Dataset):
    def __init__(self, sample: dict[str, object]) -> None:
        self.sample = sample
        self.tile_names = ["1"]
        self.requested_indices: list[int] = []

    def __len__(self) -> int:
        raise AssertionError("Whole-tile inference should iterate tile names, not dataset length")

    def get_tile(self, index: int) -> dict[str, object]:
        self.requested_indices.append(index)
        return self.sample


class MFNetTrainingTest(unittest.TestCase):
    def _make_train_spy_model(self) -> torch.nn.Module:
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.tensor(1.0))
        model.last_call = None

        def forward(self: torch.nn.Module, rgb: torch.Tensor, dsm: torch.Tensor, mode: str = "Train") -> torch.Tensor:
            self.last_call = (rgb.detach().clone(), dsm.detach().clone(), str(mode))
            return torch.stack([rgb[:, 0] - self.weight, rgb[:, 0] + self.weight], dim=1)

        model.forward = types.MethodType(forward, model)
        return model

    def _make_fixed_logits_model(self, logits: torch.Tensor) -> torch.nn.Module:
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.tensor(1.0))
        model.logits = logits.detach().clone()

        def forward(self: torch.nn.Module, rgb: torch.Tensor, dsm: torch.Tensor, mode: str = "Train") -> torch.Tensor:
            del rgb, dsm, mode
            return self.logits.to(self.weight.device) + (self.weight * 0.0)

        model.forward = types.MethodType(forward, model)
        return model

    def _write_vaihingen_sample(self, root: Path, tile_name: str = "1") -> None:
        (root / "rgb").mkdir()
        (root / "dsm").mkdir()
        (root / "labels").mkdir()
        (root / "labels_eroded").mkdir()

        rgb = torch.zeros(32, 32, 3, dtype=torch.uint8).numpy()
        rgb[:, :, 0] = 255
        dsm = (torch.arange(32 * 32, dtype=torch.int16).reshape(32, 32)).numpy()
        label = torch.zeros(32, 32, 3, dtype=torch.uint8).numpy()
        label[:, :16] = torch.tensor([255, 255, 255], dtype=torch.uint8).numpy()
        label[:, 16:] = torch.tensor([0, 0, 255], dtype=torch.uint8).numpy()
        eroded_label = torch.zeros(32, 32, 3, dtype=torch.uint8).numpy()
        eroded_label[:, :] = torch.tensor([0, 0, 255], dtype=torch.uint8).numpy()

        Image.fromarray(rgb).save(root / "rgb" / f"top_mosaic_09cm_area{tile_name}.tif")
        Image.fromarray(dsm).save(root / "dsm" / f"dsm_09cm_matching_area{tile_name}.tif")
        Image.fromarray(label).save(root / "labels" / f"top_mosaic_09cm_area{tile_name}.tif")
        Image.fromarray(eroded_label).save(root / "labels_eroded" / f"top_mosaic_09cm_area{tile_name}_noBoundary.tif")

    def _write_potsdam_sample(self, root: Path, tile_name: str = "6_10") -> None:
        (root / "rgbir").mkdir()
        (root / "dsm").mkdir()
        (root / "labels").mkdir()
        (root / "labels_eroded").mkdir()

        rgbir = torch.zeros(32, 32, 4, dtype=torch.uint8).numpy()
        rgbir[:, :, 0] = 255
        rgbir[:, :, 1] = 64
        rgbir[:, :, 2] = 32
        rgbir[:, :, 3] = 200
        dsm = (torch.arange(32 * 32, dtype=torch.uint8).reshape(32, 32)).numpy()
        label = torch.zeros(32, 32, 3, dtype=torch.uint8).numpy()
        label[:, :16] = torch.tensor([255, 255, 255], dtype=torch.uint8).numpy()
        label[:, 16:] = torch.tensor([0, 0, 255], dtype=torch.uint8).numpy()
        eroded_label = label.copy()

        Image.fromarray(rgbir).save(root / "rgbir" / f"top_potsdam_{tile_name}_RGBIR.tif")
        Image.fromarray(dsm).save(root / "dsm" / f"dsm_potsdam_{tile_name}_normalized_lastools.jpg")
        Image.fromarray(label).save(root / "labels" / f"top_potsdam_{tile_name}_label.tif")
        Image.fromarray(eroded_label).save(root / "labels_eroded" / f"top_potsdam_{tile_name}_label_noBoundary.tif")

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
            evaluator=None,
            inferencer=None,
            device=torch.device("cpu"),
            cfg={
                "work_dir": tmpdir,
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
            "meta": {"tile_name": "1"},
        }
        model = self._make_train_spy_model()
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
            "meta": {"tile_name": "1"},
        }
        model = self._make_train_spy_model()
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
            "meta": {"tile_name": "1"},
        }
        model = self._make_train_spy_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._build_trainer(tmpdir, model, sample)
            batch = next(iter(trainer.train_loader))

            with self.assertRaisesRegex(TypeError, "expected target dtype torch.long"):
                trainer.train_forward(batch)

    def test_train_forward_delegates_loss_and_metric_computation(self) -> None:
        sample = {
            "inputs": {
                "rgb": torch.rand(3, 12, 12),
                "dsm": torch.rand(12, 12),
            },
            "target": torch.randint(0, 2, (12, 12), dtype=torch.long),
            "meta": {"tile_name": "1"},
        }
        model = self._make_train_spy_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            train_loader = DataLoader(_SingleBatchDataset(sample), batch_size=1, shuffle=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            trainer = _DelegatingMFNetTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                train_loader=train_loader,
                val_loader=[],
                logger=MFNetLogger(tmpdir),
                evaluator=None,
                inferencer=None,
                device=torch.device("cpu"),
                cfg={
                    "work_dir": tmpdir,
                    "max_epochs": 1,
                    "batch_size": 1,
                    "effective_batch_size": 1,
                    "log_step_interval": 1,
                    "val_epoch_interval": 0,
                    "save_epoch_interval": 1,
                    "save_step_interval": 0,
                },
            )

            batch = next(iter(trainer.train_loader))
            loss, metrics = trainer.train_forward(batch)

            self.assertTrue(trainer.compute_loss_and_metrics_called)
            self.assertEqual(float(loss.detach()), 3.0)
            self.assertEqual(metrics, {"loss": 3.0, "accuracy": 25.0})

    def test_train_forward_uses_mfnet_ignore_loss_and_accuracy(self) -> None:
        sample = {
            "inputs": {
                "rgb": torch.rand(3, 2, 2),
                "dsm": torch.rand(2, 2),
            },
            "target": torch.tensor([[0, 255], [1, 255]], dtype=torch.long),
            "meta": {"tile_name": "1"},
        }
        logits = torch.tensor(
            [
                [
                    [[4.0, 3.0], [1.0, 0.0]],
                    [[0.0, 1.0], [5.0, 2.0]],
                ]
            ],
            dtype=torch.float32,
        )
        model = self._make_fixed_logits_model(logits)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._build_trainer(tmpdir, model, sample)
            batch = next(iter(trainer.train_loader))

            loss, metrics = trainer.train_forward(batch)

            expected_loss = torch.nn.functional.cross_entropy(
                torch.tensor([[4.0, 0.0], [1.0, 5.0]], dtype=torch.float32),
                torch.tensor([0, 1], dtype=torch.long),
            )
            self.assertAlmostEqual(float(loss.detach()), float(expected_loss), places=6)
            self.assertAlmostEqual(metrics["accuracy"], 50.0, places=6)

    def test_train_forward_returns_zero_loss_when_all_pixels_are_ignored(self) -> None:
        sample = {
            "inputs": {
                "rgb": torch.rand(3, 2, 2),
                "dsm": torch.rand(2, 2),
            },
            "target": torch.full((2, 2), 255, dtype=torch.long),
            "meta": {"tile_name": "1"},
        }
        logits = torch.tensor(
            [
                [
                    [[4.0, 3.0], [1.0, 0.0]],
                    [[0.0, 1.0], [5.0, 2.0]],
                ]
            ],
            dtype=torch.float32,
        )
        model = self._make_fixed_logits_model(logits)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._build_trainer(tmpdir, model, sample)
            batch = next(iter(trainer.train_loader))

            loss, metrics = trainer.train_forward(batch)

            self.assertEqual(float(loss.detach()), 0.0)
            self.assertEqual(metrics["accuracy"], 0.0)

    def test_data_utils_mfnet_loss_matches_ignore_label_behavior(self) -> None:
        logits = torch.tensor(
            [
                [
                    [[4.0, 3.0], [1.0, 0.0]],
                    [[0.0, 1.0], [5.0, 2.0]],
                ]
            ],
            dtype=torch.float32,
        )
        target = torch.tensor([[[0, 255], [1, 255]]], dtype=torch.long)

        loss = DataUtils.cross_entropy_filtered(logits=logits, target=target)

        expected_loss = torch.nn.functional.cross_entropy(
            torch.tensor([[4.0, 0.0], [1.0, 5.0]], dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.long),
        )
        self.assertAlmostEqual(float(loss.detach()), float(expected_loss), places=6)

    def test_normalize_dsm_matches_original_non_hunan_formula(self) -> None:
        dsm = torch.tensor([[10.0, 20.0], [30.0, 40.0]]).numpy()

        normalized = DataUtils.normalize_dsm(dsm)

        self.assertEqual(float(normalized[0, 0]), 0.0)
        self.assertEqual(float(normalized[-1, -1]), 1.0)

    def test_train_one_epoch_uses_micro_batch_steps_without_grad_accum_state(self) -> None:
        sample = {
            "inputs": {
                "rgb": torch.rand(3, 12, 12),
                "dsm": torch.rand(12, 12),
            },
            "target": torch.randint(0, 2, (12, 12), dtype=torch.long),
            "meta": {"tile_name": "1"},
        }
        model = self._make_train_spy_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            train_loader = DataLoader(_ListDataset([sample, sample]), batch_size=1, shuffle=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            trainer = MFNetTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                train_loader=train_loader,
                val_loader=[],
                logger=MFNetLogger(tmpdir),
                evaluator=None,
                inferencer=None,
                device=torch.device("cpu"),
                cfg={
                    "work_dir": tmpdir,
                    "max_epochs": 1,
                    "batch_size": 1,
                    "effective_batch_size": 2,
                    "log_step_interval": 1,
                    "val_epoch_interval": 0,
                    "save_epoch_interval": 1,
                    "save_step_interval": 0,
                },
            )

            trainer.train_one_epoch()

            self.assertEqual(trainer.global_step, 2)
            self.assertEqual(trainer.total_steps_per_epoch, 2)
            self.assertFalse(hasattr(trainer, "grad_accum_steps"))

    def test_vaihingen_dataset_emits_real_training_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)

            random.seed(0)
            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                tile_names=["1"],
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

    def test_vaihingen_dataset_augments_cropped_patch_not_full_tile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)

            random.seed(0)
            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                tile_names=["1"],
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=True,
                split="train",
            )

            sample = dataset[0]

            self.assertEqual(sample["inputs"]["rgb"].shape, (3, 16, 16))
            self.assertEqual(sample["inputs"]["dsm"].shape, (16, 16))
            self.assertEqual(sample["target"].shape, (16, 16))

    def test_vaihingen_dataset_get_tile_returns_uncropped_full_tile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)

            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                tile_names=["1"],
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=True,
                split="val",
            )

            sample = dataset.get_tile(0)

            self.assertEqual(sample["inputs"]["rgb"].shape, (3, 32, 32))
            self.assertEqual(sample["inputs"]["dsm"].shape, (32, 32))
            self.assertEqual(sample["target"].shape, (32, 32))
            self.assertEqual(sample["meta"]["tile_name"], "1")
            self.assertEqual(set(torch.unique(sample["target"]).tolist()), {1})

    def test_whole_tile_sliding_window_matches_original_mfnet_path(self) -> None:
        rgb = torch.arange(3 * 5 * 5, dtype=torch.float32).reshape(3, 5, 5)
        dsm = torch.arange(5 * 5, dtype=torch.float32).reshape(5, 5)
        target = torch.zeros(5, 5, dtype=torch.long)
        dataset = _WholeTileDataset(
            {
                "inputs": {"rgb": rgb, "dsm": dsm},
                "target": target,
                "meta": {"tile_name": "1"},
            }
        )
        modes: list[str] = []

        def model(rgb: torch.Tensor, dsm: torch.Tensor, mode: str = "Train") -> torch.Tensor:
            del dsm
            modes.append(str(mode))
            return torch.stack([torch.zeros_like(rgb[:, 0]), torch.ones_like(rgb[:, 0])], dim=1)

        outputs = SlidingWindowInferencer().run(
            model=model,
            dataset=dataset,
            device=torch.device("cpu"),
            stride=3,
            batch_size=2,
            window_size=(3, 3),
            num_classes=2,
            input_modals=("rgb", "dsm"),
            model_kwargs={"mode": "Test"},
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["pred"].shape, (5, 5))
        self.assertTrue(torch.equal(outputs[0]["pred"], torch.ones(5, 5, dtype=torch.long)))
        self.assertIs(outputs[0]["target"], target)
        self.assertEqual(outputs[0]["meta"], {"tile_name": "1"})
        self.assertEqual(dataset.requested_indices, [0])
        self.assertTrue(modes)
        self.assertTrue(all(mode == "Test" for mode in modes))

    def test_whole_tile_sliding_window_uses_explicit_rgb_dsm_order(self) -> None:
        rgb = torch.arange(3 * 5 * 5, dtype=torch.float32).reshape(3, 5, 5)
        dsm = torch.arange(5 * 5, dtype=torch.float32).reshape(5, 5)
        target = torch.zeros(5, 5, dtype=torch.long)
        dataset = _WholeTileDataset(
            {
                "inputs": {"dsm": dsm, "rgb": rgb},
                "target": target,
                "meta": {"tile_name": "1"},
            }
        )
        modes: list[str] = []
        call_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def model(rgb: torch.Tensor, dsm: torch.Tensor, mode: str = "Train") -> torch.Tensor:
            modes.append(str(mode))
            call_shapes.append((tuple(rgb.shape), tuple(dsm.shape)))
            return torch.stack([torch.zeros_like(rgb[:, 0]), torch.ones_like(rgb[:, 0])], dim=1)

        SlidingWindowInferencer().run(
            model=model,
            dataset=dataset,
            device=torch.device("cpu"),
            stride=3,
            batch_size=2,
            window_size=(3, 3),
            num_classes=2,
            input_modals=("rgb", "dsm"),
            model_kwargs={"mode": "Test"},
        )

        self.assertTrue(modes)
        self.assertTrue(all(mode == "Test" for mode in modes))
        self.assertEqual(call_shapes[0], ((2, 3, 3, 3), (2, 3, 3)))

    def test_whole_tile_sliding_window_accepts_generic_multimodal_inputs(self) -> None:
        rgb = torch.arange(3 * 5 * 5, dtype=torch.float32).reshape(3, 5, 5)
        dsm = torch.arange(5 * 5, dtype=torch.float32).reshape(5, 5)
        target = torch.zeros(5, 5, dtype=torch.long)
        dataset = _WholeTileDataset(
            {
                "inputs": {"rgb": rgb, "dsm": dsm},
                "target": target,
                "meta": {"tile_name": "1"},
            }
        )
        modes: list[str] = []

        def model(rgb: torch.Tensor, dsm: torch.Tensor, mode: str = "Train") -> torch.Tensor:
            del dsm
            modes.append(str(mode))
            return torch.stack([torch.zeros_like(rgb[:, 0]), torch.ones_like(rgb[:, 0])], dim=1)

        outputs = SlidingWindowInferencer().run(
            model=model,
            dataset=dataset,
            device=torch.device("cpu"),
            stride=3,
            batch_size=2,
            window_size=(3, 3),
            num_classes=2,
            input_modals=("rgb", "dsm"),
            model_kwargs={"mode": "Test"},
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["pred"].shape, (5, 5))
        self.assertTrue(torch.equal(outputs[0]["pred"], torch.ones(5, 5, dtype=torch.long)))
        self.assertIs(outputs[0]["target"], target)
        self.assertEqual(outputs[0]["meta"], {"tile_name": "1"})
        self.assertEqual(dataset.requested_indices, [0])
        self.assertTrue(modes)
        self.assertTrue(all(mode == "Test" for mode in modes))

    def test_whole_tile_sliding_window_accepts_single_input_key(self) -> None:
        image = torch.arange(3 * 5 * 5, dtype=torch.float32).reshape(3, 5, 5)
        target = torch.zeros(5, 5, dtype=torch.long)
        dataset = _WholeTileDataset(
            {
                "inputs": {"image": image},
                "target": target,
                "meta": {"tile_name": "1"},
            }
        )
        batch_shapes: list[tuple[int, ...]] = []

        def model(image: torch.Tensor) -> torch.Tensor:
            batch_shapes.append(tuple(image.shape))
            return torch.stack([torch.zeros_like(image[:, 0]), torch.ones_like(image[:, 0])], dim=1)

        outputs = SlidingWindowInferencer().run(
            model=model,
            dataset=dataset,
            device=torch.device("cpu"),
            stride=3,
            batch_size=2,
            window_size=(3, 3),
            num_classes=2,
            input_modals=("image",),
        )

        self.assertEqual(len(outputs), 1)
        self.assertTrue(torch.equal(outputs[0]["pred"], torch.ones(5, 5, dtype=torch.long)))
        self.assertIs(outputs[0]["target"], target)
        self.assertEqual(outputs[0]["meta"], {"tile_name": "1"})
        self.assertEqual(dataset.requested_indices, [0])
        self.assertEqual(batch_shapes[0], (2, 3, 3, 3))

    def test_potsdam_dataset_emits_same_training_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_potsdam_sample(root)

            random.seed(0)
            dataset = ISPRSDataset(
                preset=POTSDAM_PRESET,
                root_dir=str(root),
                tile_names=["6_10"],
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

    def test_training_crop_bounds_match_original_random_pos(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)
            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                tile_names=["1"],
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=False,
                split="train",
            )

            with patch("data.isprs_dataset.random.randint", side_effect=[3, 5]) as randint:
                sample = dataset[0]

            self.assertEqual(randint.call_args_list, [call(0, 15), call(0, 15)])
            self.assertEqual(sample["inputs"]["rgb"].shape, (3, 16, 16))
            self.assertEqual(sample["inputs"]["dsm"].shape, (16, 16))

    def test_build_isprs_dataset_dispatches_to_potsdam(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_potsdam_sample(root)

            dataset = build_isprs_dataset(
                "potsdam",
                root_dir=str(root),
                tile_names=["6_10"],
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=False,
                split="val",
            )

            self.assertIsInstance(dataset, ISPRSDataset)
            self.assertEqual(dataset.preset.name, "potsdam")
            sample = dataset[0]
            self.assertEqual(sample["inputs"]["rgb"].shape, (3, 16, 16))

    def test_build_model_passes_sam_checkpoint_to_mfnet(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormer:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormer = FakeUNetFormer
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer",
                    "num_classes": 6,
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormer)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def _load_train_entry_module(self):
        spec = importlib.util.spec_from_file_location(
            "test_train_module",
            Path(__file__).resolve().parents[1] / "train.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _make_train_entry_config(self, root_dir: str) -> dict[str, object]:
        return {
            "model": {
                "type": "mfnet_unetformer",
                "num_classes": 6,
                "sam_checkpoint": "/tmp/sam_vit_b.pth",
            },
            "dataset": {
                "name": "vaihingen",
                "root_dir": root_dir,
                "patch_size": [32, 32],
                "train_tile_names": ["1"],
                "val_tile_names": ["5"],
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
                "weight_decay": 0.0005,
            },
            "scheduler": {"type": "MultiStepLR", "milestones": [1, 2], "gamma": 0.1},
            "validation": {
                "stride": 32,
            },
            "train": {
                "max_epochs": 1,
                "batch_size": 2,
                "auto_resume": True,
                "log_step_interval": 1,
                "val_epoch_interval": 1,
                "save_epoch_interval": 1,
                "save_step_interval": 0,
                "use_tensorboard": True,
            },
        }

    def _run_train_entry(
        self,
        module,
        args: object,
        cfg: dict[str, object],
        default_work_dir: Path | None,
    ) -> dict[str, object]:
        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.image_encoder = torch.nn.Linear(1, 1, bias=False)

        captured_dataset_calls: list[dict[str, object]] = []
        captured_trainer_kwargs: list[dict[str, object]] = []
        captured_model_cfg: list[dict[str, object]] = []
        captured_load_config_paths: list[str] = []
        captured_default_work_dir_calls: list[dict[str, object]] = []

        class FakeTrainer:
            def __init__(self, **kwargs: object) -> None:
                captured_trainer_kwargs.append(kwargs)

            def train(self) -> None:
                return None

        original_parse_args = module.parse_args
        original_load_config = module.load_config
        original_build_default_work_dir = module.build_default_work_dir
        original_build_model = module.build_model
        original_build_isprs_dataset = module.build_isprs_dataset
        original_dataloader = module.DataLoader
        original_trainer = module.MFNetTrainer
        try:
            module.parse_args = lambda: args

            def fake_load_config(path: str) -> dict[str, object]:
                captured_load_config_paths.append(path)
                return cfg

            module.load_config = fake_load_config

            def fake_build_default_work_dir(
                model_name: object,
                dataset_name: object,
                root_dir: str | Path = "work_dirs",
            ) -> Path:
                captured_default_work_dir_calls.append(
                    {
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "root_dir": root_dir,
                    }
                )
                if default_work_dir is None:
                    raise AssertionError("resume-dir should not build a new workdir")
                return default_work_dir

            module.build_default_work_dir = fake_build_default_work_dir

            def fake_build_model(model_cfg: dict[str, object]) -> FakeModel:
                captured_model_cfg.append(model_cfg)
                return FakeModel()

            module.build_model = fake_build_model

            def fake_build_isprs_dataset(name: str, **kwargs: object) -> dict[str, object]:
                captured_dataset_calls.append({"name": name, **kwargs})
                return {"dataset_name": name, "dataset_kwargs": kwargs}

            module.build_isprs_dataset = fake_build_isprs_dataset
            module.DataLoader = lambda dataset, **kwargs: {
                "dataset": dataset,
                "batch_size": kwargs["batch_size"],
                "loader_kwargs": kwargs,
            }
            module.MFNetTrainer = FakeTrainer

            module.main()
        finally:
            module.parse_args = original_parse_args
            module.load_config = original_load_config
            module.build_default_work_dir = original_build_default_work_dir
            module.build_model = original_build_model
            module.build_isprs_dataset = original_build_isprs_dataset
            module.DataLoader = original_dataloader
            module.MFNetTrainer = original_trainer

        return {
            "dataset_calls": captured_dataset_calls,
            "trainer_kwargs": captured_trainer_kwargs,
            "model_cfg": captured_model_cfg,
            "load_config_paths": captured_load_config_paths,
            "default_work_dir_calls": captured_default_work_dir_calls,
        }

    def test_train_default_work_dir_uses_model_dataset_and_timestamp(self) -> None:
        module = self._load_train_entry_module()

        with patch.object(sys, "argv", ["train.py"]):
            args = module.parse_args()

        self.assertFalse(hasattr(args, "work_dir"))
        self.assertFalse(hasattr(args, "resume_from"))
        self.assertIsNone(args.resume_dir)
        self.assertIsNone(args.resume_ckpt)

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = module.build_default_work_dir(
                model_name="MFNet UNetFormer",
                dataset_name="Potsdam/RGB",
                root_dir=tmpdir,
            )

            self.assertEqual(work_dir.parent, Path(tmpdir))
            self.assertRegex(
                work_dir.name,
                re.compile(r"^mfnet_unetformer_potsdam_rgb_\d{6}_\d{6}$"),
            )

    def test_train_entry_builds_mfnet_trainer_with_sgd_and_scheduler(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            config_path = config_dir / "external_config.jsonc"
            config_text = '{"train": {"max_epochs": 1}}\n'
            config_path.write_text(config_text, encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": "/tmp/pretrained.pth",
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=self._make_train_entry_config(root_dir=str(work_dir)),
                default_work_dir=work_dir,
            )

            dataset_calls = result["dataset_calls"]
            self.assertEqual(len(dataset_calls), 2)
            self.assertEqual(dataset_calls[0]["name"], "vaihingen")
            self.assertEqual(dataset_calls[1]["name"], "vaihingen")
            self.assertEqual(dataset_calls[0]["split"], "train")
            self.assertEqual(dataset_calls[1]["split"], "val")
            self.assertEqual(dataset_calls[0]["tile_names"], ["1"])
            self.assertEqual(dataset_calls[1]["tile_names"], ["5"])
            self.assertEqual(result["load_config_paths"], [str(config_path)])
            self.assertEqual(len(result["default_work_dir_calls"]), 1)
            self.assertEqual(len(result["trainer_kwargs"]), 1)
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["optimizer"], torch.optim.SGD)
            self.assertIsInstance(
                trainer_kwargs["scheduler"],
                torch.optim.lr_scheduler.MultiStepLR,
            )
            self.assertEqual(len(result["model_cfg"]), 1)
            self.assertEqual(result["model_cfg"][0]["sam_checkpoint"], "/tmp/sam_vit_b.pth")
            self.assertIsInstance(trainer_kwargs["logger"], MFNetLogger)
            self.assertTrue(trainer_kwargs["logger"].use_tensorboard)
            self.assertEqual(trainer_kwargs["cfg"]["val_epoch_interval"], 1)
            self.assertEqual(trainer_kwargs["cfg"]["batch_size"], 2)
            self.assertEqual(trainer_kwargs["cfg"]["experiment_name"], work_dir.name)
            self.assertEqual(trainer_kwargs["cfg"]["sam_checkpoint"], "/tmp/sam_vit_b.pth")
            self.assertIsNone(trainer_kwargs["cfg"]["resume_from"])
            self.assertEqual(trainer_kwargs["cfg"]["load_from"], "/tmp/pretrained.pth")
            self.assertEqual(trainer_kwargs["cfg"]["work_dir"], str(work_dir))
            self.assertNotIn("checkpoint_manager", trainer_kwargs)
            self.assertEqual(
                trainer_kwargs["cfg"]["validation"],
                {"stride": 32},
            )
            self.assertNotIn("effective_batch_size", trainer_kwargs["cfg"])
            self.assertEqual((work_dir / config_path.name).read_text(encoding="utf-8"), config_text)
            self.assertFalse((work_dir / "train_config.jsonc").exists())

    def test_train_entry_uses_resume_ckpt_for_new_experiment_only(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            resume_ckpt = Path(tmpdir) / "manual_resume.pth"
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": str(resume_ckpt),
                    "load_from": "/tmp/pretrained.pth",
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=self._make_train_entry_config(root_dir=str(work_dir)),
                default_work_dir=work_dir,
            )

            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertEqual(result["load_config_paths"], [str(config_path)])
            self.assertEqual(trainer_kwargs["cfg"]["work_dir"], str(work_dir))
            self.assertEqual(trainer_kwargs["cfg"]["resume_from"], str(resume_ckpt))
            self.assertEqual(trainer_kwargs["cfg"]["load_from"], "/tmp/pretrained.pth")
            self.assertTrue((work_dir / config_path.name).is_file())

    def test_train_entry_resume_dir_overrides_file_parameters(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            resume_dir = Path(tmpdir) / "resume_work"
            resume_dir.mkdir()
            resume_config_path = resume_dir / "resume_config.json"
            resume_config_text = '{"train": {"max_epochs": 9}}\n'
            resume_config_path.write_text(resume_config_text, encoding="utf-8")
            external_config_path = Path(tmpdir) / "external_config.jsonc"
            external_config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "config": str(external_config_path),
                    "device": "cpu",
                    "resume_dir": str(resume_dir),
                    "resume_ckpt": str(Path(tmpdir) / "manual_resume.pth"),
                    "load_from": "/tmp/pretrained.pth",
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=self._make_train_entry_config(root_dir=str(resume_dir)),
                default_work_dir=None,
            )

            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertEqual(result["load_config_paths"], [str(resume_config_path)])
            self.assertEqual(result["default_work_dir_calls"], [])
            self.assertEqual(trainer_kwargs["cfg"]["work_dir"], str(resume_dir))
            self.assertEqual(trainer_kwargs["cfg"]["experiment_name"], resume_dir.name)
            self.assertEqual(trainer_kwargs["cfg"]["resume_from"], str(resume_dir / "latest.pth"))
            self.assertIsNone(trainer_kwargs["cfg"]["load_from"])
            self.assertEqual(resume_config_path.read_text(encoding="utf-8"), resume_config_text)
            self.assertFalse((resume_dir / external_config_path.name).exists())

if __name__ == "__main__":
    unittest.main()
