from __future__ import annotations

import tempfile
import types
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from engine import MFNetTrainer
from multimodal_helpers import _ListDataset, _SingleBatchDataset
from utils import DataUtils, IntermediateStatsRecorder, TestNetLogger as _TestNetLogger


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


class MFNetTrainerContractTest(unittest.TestCase):
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
            logger=_TestNetLogger(tmpdir),
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
            "meta": {"tile_id": "1"},
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
            "meta": {"tile_id": "1"},
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
            "meta": {"tile_id": "1"},
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
            "meta": {"tile_id": "1"},
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
                logger=_TestNetLogger(tmpdir),
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

    def test_run_train_forward_merges_intermediate_stats_and_resets_recorder(self) -> None:
        sample = {
            "inputs": {
                "rgb": torch.rand(3, 12, 12),
                "dsm": torch.rand(12, 12),
            },
            "target": torch.randint(0, 2, (12, 12), dtype=torch.long),
            "meta": {"tile_id": "1"},
        }
        model = self._make_train_spy_model()
        model.intermediate_stats = IntermediateStatsRecorder()
        model.intermediate_stats.record_scalar("stale/value", 1.0)

        def forward(self: torch.nn.Module, rgb: torch.Tensor, dsm: torch.Tensor, mode: str = "Train") -> torch.Tensor:
            del dsm, mode
            self.intermediate_stats.record_scalar("fresh/value", 7.0)
            return torch.stack([rgb[:, 0] - self.weight, rgb[:, 0] + self.weight], dim=1)

        model.forward = types.MethodType(forward, model)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._build_trainer(tmpdir, model, sample)
            batch = next(iter(trainer.train_loader))

            loss, metrics = trainer._run_train_forward(batch)

            self.assertEqual(loss.ndim, 0)
            self.assertEqual(metrics["fresh/value"], 7.0)
            self.assertNotIn("stale/value", metrics)
            self.assertEqual(model.intermediate_stats.snapshot(), {})

    def test_train_forward_uses_mfnet_ignore_loss_and_accuracy(self) -> None:
        sample = {
            "inputs": {
                "rgb": torch.rand(3, 2, 2),
                "dsm": torch.rand(2, 2),
            },
            "target": torch.tensor([[0, 255], [1, 255]], dtype=torch.long),
            "meta": {"tile_id": "1"},
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
            "meta": {"tile_id": "1"},
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

    def test_enhance_dsm_similarity_returns_zero_for_constant_dsm(self) -> None:
        dsm = np.full((5, 6), 42.0, dtype=np.float32)

        enhanced = DataUtils.enhance_dsm_similarity(dsm)

        self.assertEqual(enhanced.shape, dsm.shape)
        self.assertEqual(enhanced.dtype, np.float32)
        self.assertTrue(np.isfinite(enhanced).all())
        self.assertTrue(np.array_equal(enhanced, np.zeros_like(dsm, dtype=np.float32)))

    def test_enhance_dsm_similarity_preserves_single_channel_contract(self) -> None:
        dsm = np.arange(8 * 9, dtype=np.float32).reshape(8, 9)

        enhanced = DataUtils.enhance_dsm_similarity(dsm)

        self.assertEqual(enhanced.shape, (8, 9))
        self.assertEqual(enhanced.dtype, np.float32)
        self.assertTrue(np.isfinite(enhanced).all())
        self.assertGreaterEqual(float(enhanced.min()), 0.0)
        self.assertLessEqual(float(enhanced.max()), 1.0)

    def test_enhance_dsm_similarity_matches_reference_loop(self) -> None:
        dsm = np.array(
            [
                [1.0, 2.0, 4.0, 8.0],
                [3.0, 5.0, 7.0, 9.0],
                [2.0, 6.0, 10.0, 12.0],
                [4.0, 8.0, 11.0, 13.0],
            ],
            dtype=np.float32,
        )
        window_size = 3
        sigma = 0.2
        lambda_weight = 0.3

        enhanced = DataUtils.enhance_dsm_similarity(
            dsm,
            window_size=window_size,
            sigma=sigma,
            lambda_weight=lambda_weight,
        )

        normalized = (dsm - float(dsm.min())) / (float(dsm.max()) - float(dsm.min()))
        padded = np.pad(normalized, pad_width=window_size // 2, mode="reflect")
        expected = np.zeros_like(normalized, dtype=np.float32)
        for row in range(normalized.shape[0]):
            for col in range(normalized.shape[1]):
                window = padded[row : row + window_size, col : col + window_size]
                weights = np.exp(-((window - normalized[row, col]) ** 2) / (2.0 * sigma**2))
                similarity = float(np.mean(weights))
                expected[row, col] = (
                    normalized[row, col] + lambda_weight * normalized[row, col] * similarity
                ) / (1.0 + lambda_weight)

        np.testing.assert_allclose(enhanced, expected, rtol=1e-6, atol=1e-6)

    def test_enhance_dsm_similarity_rejects_invalid_parameters(self) -> None:
        dsm = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)

        with self.assertRaisesRegex(ValueError, "positive odd integer"):
            DataUtils.enhance_dsm_similarity(dsm, window_size=0)
        with self.assertRaisesRegex(ValueError, "positive odd integer"):
            DataUtils.enhance_dsm_similarity(dsm, window_size=4)
        with self.assertRaisesRegex(ValueError, "sigma to be positive"):
            DataUtils.enhance_dsm_similarity(dsm, sigma=0.0)

    def test_train_one_epoch_uses_micro_batch_steps_without_grad_accum_state(self) -> None:
        sample = {
            "inputs": {
                "rgb": torch.rand(3, 12, 12),
                "dsm": torch.rand(12, 12),
            },
            "target": torch.randint(0, 2, (12, 12), dtype=torch.long),
            "meta": {"tile_id": "1"},
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
                logger=_TestNetLogger(tmpdir),
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
