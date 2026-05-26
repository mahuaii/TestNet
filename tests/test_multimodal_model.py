from __future__ import annotations

import importlib
import importlib.util
import json
import random
import re
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
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
from utils import DataUtils, IntermediateStatsRecorder, TestNetRecorderLogger, TestNetLogger


@contextmanager
def _fake_mfnet_optional_imports() -> object:
    fake_timm_module = types.ModuleType("timm")
    fake_timm_models_module = types.ModuleType("timm.models")
    fake_timm_layers_module = types.ModuleType("timm.models.layers")
    fake_timm_layers_module.DropPath = torch.nn.Identity
    fake_timm_layers_module.to_2tuple = lambda value: (value, value)
    fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
    fake_cv2_module = types.ModuleType("cv2")
    fake_cv2_module.COLORMAP_JET = 2
    fake_cv2_module.applyColorMap = lambda image, color_map: image
    fake_cv2_module.imwrite = lambda path, image: True
    originals = {
        "timm": sys.modules.get("timm"),
        "timm.models": sys.modules.get("timm.models"),
        "timm.models.layers": sys.modules.get("timm.models.layers"),
        "cv2": sys.modules.get("cv2"),
    }
    try:
        sys.modules["timm"] = fake_timm_module
        sys.modules["timm.models"] = fake_timm_models_module
        sys.modules["timm.models.layers"] = fake_timm_layers_module
        sys.modules["cv2"] = fake_cv2_module
        yield
    finally:
        for module_name, original in originals.items():
            if original is None:
                del sys.modules[module_name]
            else:
                sys.modules[module_name] = original


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


class MMAdapter10FusionBlockTest(unittest.TestCase):
    def _make_block(self) -> torch.nn.Module:
        with _fake_mfnet_optional_imports():
            from models.mfnet.sam_adapted.ImageEncoder.vit.adapter_fusionblock import MMAdapter10FusionBlock

            args = types.SimpleNamespace(mid_dim=None)
            return MMAdapter10FusionBlock(
                args=args,
                dim=8,
                num_heads=2,
                mlp_ratio=2.0,
                qkv_bias=True,
                window_size=0,
            )

    def test_patch_wise_fusion_uses_patch_scalar_gates(self) -> None:
        block = self._make_block()
        x_msg = torch.randn(2, 3, 5, 8)
        y_msg = torch.randn(2, 3, 5, 8)

        x_fuse, y_fuse = block.fuse_adapter_messages(x_msg, y_msg)
        gate = block.MMAdapter_Fusion(torch.cat([x_msg, y_msg], dim=-1))
        gate_y_to_x = gate[..., 0:1]
        gate_x_to_y = gate[..., 1:2]

        self.assertEqual(x_fuse.shape, x_msg.shape)
        self.assertEqual(y_fuse.shape, y_msg.shape)
        self.assertEqual(gate.shape, (2, 3, 5, 2))
        self.assertEqual(gate_y_to_x.shape, (2, 3, 5, 1))
        self.assertEqual(gate_x_to_y.shape, (2, 3, 5, 1))
        self.assertTrue(torch.allclose(x_fuse, x_msg + gate_y_to_x * y_msg))
        self.assertTrue(torch.allclose(y_fuse, y_msg + gate_x_to_y * x_msg))

    def test_patch_wise_fusion_rejects_non_spatial_tokens(self) -> None:
        block = self._make_block()

        with self.assertRaisesRegex(ValueError, r"\[B, H, W, C\]"):
            block.fuse_adapter_messages(torch.randn(2, 15, 8), torch.randn(2, 15, 8))

    def test_mmadapter10_block_forward_and_parameters(self) -> None:
        block = self._make_block()
        x = torch.randn(2, 3, 5, 8)
        y = torch.randn(2, 3, 5, 8)

        out_x, out_y = block(x, y)
        named_parameters = dict(block.named_parameters())

        self.assertEqual(out_x.shape, x.shape)
        self.assertEqual(out_y.shape, y.shape)
        self.assertNotIn("wx_Adapter", named_parameters)
        self.assertNotIn("wy_Adapter", named_parameters)
        fusion_parameter_names = [
            name for name in named_parameters if name.startswith("MMAdapter_Fusion.")
        ]
        self.assertTrue(fusion_parameter_names)
        self.assertTrue(all(named_parameters[name].requires_grad for name in fusion_parameter_names))

    def test_image_encoder_uses_mmadapter10_block_when_args_select_it(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.sam_adapted.ImageEncoder.vit.adapter_fusionblock import MMAdapter10FusionBlock
            from models.mfnet.sam_adapted.sam.modeling.image_encoder import ImageEncoderViT

            args = types.SimpleNamespace(mod="sam_adpt", mid_dim=None, mm_adapter_block="mmadapter10")
            encoder = ImageEncoderViT(
                args=args,
                img_size=16,
                patch_size=16,
                embed_dim=8,
                depth=1,
                num_heads=2,
                out_chans=4,
                use_abs_pos=False,
                use_rel_pos=False,
            )

        self.assertIsInstance(encoder.blocks[0], MMAdapter10FusionBlock)

    def test_image_encoder_uses_default_adapter_fusion_block_without_mmadapter10(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.sam_adapted.ImageEncoder.vit.adapter_fusionblock import AdapterFusionBlock
            from models.mfnet.sam_adapted.sam.modeling.image_encoder import ImageEncoderViT

            args = types.SimpleNamespace(mod="sam_adpt", mid_dim=None)
            encoder = ImageEncoderViT(
                args=args,
                img_size=16,
                patch_size=16,
                embed_dim=8,
                depth=1,
                num_heads=2,
                out_chans=4,
                use_abs_pos=False,
                use_rel_pos=False,
            )

        self.assertIsInstance(encoder.blocks[0], AdapterFusionBlock)

    def test_unetformer_mmadapter10_passes_mmadapter10_arg_to_sam_builder(self) -> None:
        with _fake_mfnet_optional_imports():
            import models.mfnet.UNetFormer_MMSAM as module

            captured_args: list[object] = []

            class FakeSam(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.image_encoder = torch.nn.Module()

            def fake_build_sam(args: object, checkpoint: str) -> FakeSam:
                del checkpoint
                captured_args.append(args)
                return FakeSam()

            original_parse_args = module.cfg.parse_args
            original_builder = module.sam_model_registry.get("vit_b")
            try:
                module.cfg.parse_args = lambda: types.SimpleNamespace(mod="sam_adpt")
                module.sam_model_registry["vit_b"] = fake_build_sam

                module.UNetFormerMMAdapter10(
                    num_classes=6,
                    sam_backbone="vit_b",
                    sam_checkpoint="/tmp/sam_vit_b_01ec64.pth",
                )
            finally:
                module.cfg.parse_args = original_parse_args
                if original_builder is None:
                    del module.sam_model_registry["vit_b"]
                else:
                    module.sam_model_registry["vit_b"] = original_builder

        self.assertEqual(len(captured_args), 1)
        self.assertEqual(captured_args[0].mm_adapter_block, "mmadapter10")

    def test_unetformer_prealign_mmadapter10_passes_mmadapter10_arg_and_reuses_prealign_forward(self) -> None:
        with _fake_mfnet_optional_imports():
            import models.mfnet.UNetFormer_MMSAM as base_module
            from models.mfnet.UNetFormer_MMSAM_prealign import UNetFormerPreAlign
            from models.mfnet.UNetFormer_MMSAM_prealign_mmadapter10 import UNetFormerPreAlignMMAdapter10

            captured_args: list[object] = []

            class FakeSam(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.image_encoder = torch.nn.Module()

            def fake_build_sam(args: object, checkpoint: str) -> FakeSam:
                del checkpoint
                captured_args.append(args)
                return FakeSam()

            original_parse_args = base_module.cfg.parse_args
            original_builder = base_module.sam_model_registry.get("vit_b")
            try:
                base_module.cfg.parse_args = lambda: types.SimpleNamespace(mod="sam_adpt")
                base_module.sam_model_registry["vit_b"] = fake_build_sam

                UNetFormerPreAlignMMAdapter10(
                    num_classes=6,
                    sam_backbone="vit_b",
                    sam_checkpoint="/tmp/sam_vit_b_01ec64.pth",
                )
            finally:
                base_module.cfg.parse_args = original_parse_args
                if original_builder is None:
                    del base_module.sam_model_registry["vit_b"]
                else:
                    base_module.sam_model_registry["vit_b"] = original_builder

        self.assertTrue(issubclass(UNetFormerPreAlignMMAdapter10, UNetFormerPreAlign))
        self.assertNotIn("forward", UNetFormerPreAlignMMAdapter10.__dict__)
        self.assertIs(UNetFormerPreAlignMMAdapter10.forward, UNetFormerPreAlign.forward)
        self.assertEqual(len(captured_args), 1)
        self.assertEqual(captured_args[0].mm_adapter_block, "mmadapter10")


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
        self.ids = ["1"]
        self.requested_indices: list[int] = []

    def __len__(self) -> int:
        raise AssertionError("Whole-tile inference should iterate tile ids, not dataset length")

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

    def _write_vaihingen_sample(self, root: Path, tile_id: str = "1") -> None:
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

        Image.fromarray(rgb).save(root / "rgb" / f"top_mosaic_09cm_area{tile_id}.tif")
        Image.fromarray(dsm).save(root / "dsm" / f"dsm_09cm_matching_area{tile_id}.tif")
        Image.fromarray(label).save(root / "labels" / f"top_mosaic_09cm_area{tile_id}.tif")
        Image.fromarray(eroded_label).save(root / "labels_eroded" / f"top_mosaic_09cm_area{tile_id}_noBoundary.tif")

    def _write_potsdam_sample(self, root: Path, tile_id: str = "6_10") -> None:
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

        Image.fromarray(rgbir).save(root / "rgbir" / f"top_potsdam_{tile_id}_RGBIR.tif")
        Image.fromarray(dsm).save(root / "dsm" / f"dsm_potsdam_{tile_id}_normalized_lastools.jpg")
        Image.fromarray(label).save(root / "labels" / f"top_potsdam_{tile_id}_label.tif")
        Image.fromarray(eroded_label).save(root / "labels_eroded" / f"top_potsdam_{tile_id}_label_noBoundary.tif")

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
            logger=TestNetLogger(tmpdir),
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
                logger=TestNetLogger(tmpdir),
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
                logger=TestNetLogger(tmpdir),
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

    def test_vaihingen_dataset_augments_cropped_patch_not_full_tile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)

            random.seed(0)
            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                ids=["1"],
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
                ids=["1"],
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
            self.assertEqual(sample["meta"]["tile_id"], "1")
            self.assertEqual(set(torch.unique(sample["target"]).tolist()), {1})

    def test_whole_tile_sliding_window_matches_original_mfnet_path(self) -> None:
        rgb = torch.arange(3 * 5 * 5, dtype=torch.float32).reshape(3, 5, 5)
        dsm = torch.arange(5 * 5, dtype=torch.float32).reshape(5, 5)
        target = torch.zeros(5, 5, dtype=torch.long)
        dataset = _WholeTileDataset(
            {
                "inputs": {"rgb": rgb, "dsm": dsm},
                "target": target,
                "meta": {"tile_id": "1"},
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
        self.assertEqual(outputs[0]["meta"], {"tile_id": "1"})
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
                "meta": {"tile_id": "1"},
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
                "meta": {"tile_id": "1"},
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
        self.assertEqual(outputs[0]["meta"], {"tile_id": "1"})
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
                "meta": {"tile_id": "1"},
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
        self.assertEqual(outputs[0]["meta"], {"tile_id": "1"})
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
                ids=["6_10"],
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
                ids=["1"],
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
                ids=["6_10"],
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
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormer)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_unetformer_mmadapter10(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerMMAdapter10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerMMAdapter10 = FakeUNetFormerMMAdapter10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer_mmadapter10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerMMAdapter10)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_dga(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGA10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGA10 = FakeUNetFormerDGA10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer_dga10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGA10)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": False,
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_dga2(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGA20:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGA20 = FakeUNetFormerDGA20
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer_dga20",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGA20)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": False,
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_dga20_dgsf10(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGA20DGSF10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGA20DGSF10 = FakeUNetFormerDGA20DGSF10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer_dga20_dgsf10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGA20DGSF10)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": True,
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_dgsf10(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGSF10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGSF10 = FakeUNetFormerDGSF10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer_dgsf10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["dgsf10"],
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGSF10)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": True,
                        "record_intermediate_modules": ["dgsf10"],
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_prealign_auxalign_dgsf10(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerPreAlignAuxAlignDGSF10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerPreAlignAuxAlignDGSF10 = FakeUNetFormerPreAlignAuxAlignDGSF10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer_prealign_auxalign_dgsf10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["dgsf10"],
                }
            )

            self.assertIsInstance(model, FakeUNetFormerPreAlignAuxAlignDGSF10)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": True,
                        "record_intermediate_modules": ["dgsf10"],
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_passes_dga_intermediate_stats_flag(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGA10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGA10 = FakeUNetFormerDGA10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer_dga10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGA10)
            self.assertTrue(captured_kwargs[0]["record_intermediate_stats"])
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_passes_record_intermediate_modules_when_configured(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGA20DGSF10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGA20DGSF10 = FakeUNetFormerDGA20DGSF10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer_dga20_dgsf10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["dgsf10"],
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGA20DGSF10)
            self.assertEqual(captured_kwargs[0]["record_intermediate_modules"], ["dgsf10"])
            self.assertTrue(captured_kwargs[0]["record_intermediate_stats"])
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_independent_dga_softplus(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGA10Softplus:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGA10Softplus = FakeUNetFormerDGA10Softplus
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer_dga10_softplus",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGA10Softplus)
            self.assertTrue(captured_kwargs[0]["record_intermediate_stats"])
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_rejects_retired_dga_type_names(self) -> None:
        build_module = importlib.import_module("models.build")
        base_cfg = {
            "num_classes": 6,
            "sam_backbone": "vit_b",
            "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
        }

        for model_type in [
            "mfnet_unetformer_dga",
            "mfnet_unetformer_dga2",
            "mfnet_unetformer_dga3",
            "mfnet_unetformer_dga10_contrib_stats",
            "mfnet_unetformer_dga20_contrib_stats",
            "mfnet_unetformer_prealign_dga",
        ]:
            with self.subTest(model_type=model_type):
                with self.assertRaises(KeyError):
                    build_module.build_model({"type": model_type, **base_cfg})

    def test_build_model_dispatches_to_mfnet_dga30(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGA30:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGA30 = FakeUNetFormerDGA30
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer_dga30",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGA30)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_prealign(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerPreAlign:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerPreAlign = FakeUNetFormerPreAlign
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer_prealign",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerPreAlign)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_prealign_mmadapter10(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerPreAlignMMAdapter10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerPreAlignMMAdapter10 = FakeUNetFormerPreAlignMMAdapter10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer_prealign_mmadapter10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerPreAlignMMAdapter10)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_prealign_auxalign(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerPreAlignAuxAlign:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerPreAlignAuxAlign = FakeUNetFormerPreAlignAuxAlign
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer_prealign_auxalign",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerPreAlignAuxAlign)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_prealign_dga(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerPreAlignDGA10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerPreAlignDGA10 = FakeUNetFormerPreAlignDGA10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer_prealign_dga10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerPreAlignDGA10)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_prealign_auxalign_dga(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerPreAlignAuxAlignDGA10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerPreAlignAuxAlignDGA10 = FakeUNetFormerPreAlignAuxAlignDGA10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer_prealign_auxalign_dga10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerPreAlignAuxAlignDGA10)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_unetformer_prealign_expands_auxiliary_input_before_encoder(self) -> None:
        fake_timm_module = types.ModuleType("timm")
        fake_timm_models_module = types.ModuleType("timm.models")
        fake_timm_layers_module = types.ModuleType("timm.models.layers")
        fake_timm_layers_module.DropPath = torch.nn.Identity
        fake_timm_layers_module.to_2tuple = lambda value: (value, value)
        fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
        fake_cv2_module = types.ModuleType("cv2")
        fake_cv2_module.COLORMAP_JET = 2
        fake_cv2_module.applyColorMap = lambda image, color_map: image
        fake_cv2_module.imwrite = lambda path, image: True
        original_timm_module = sys.modules.get("timm")
        original_timm_models_module = sys.modules.get("timm.models")
        original_timm_layers_module = sys.modules.get("timm.models.layers")
        original_cv2_module = sys.modules.get("cv2")

        try:
            sys.modules["timm"] = fake_timm_module
            sys.modules["timm.models"] = fake_timm_models_module
            sys.modules["timm.models.layers"] = fake_timm_layers_module
            sys.modules["cv2"] = fake_cv2_module
            from models.mfnet.UNetFormer_MMSAM_prealign import UNetFormerPreAlign
        finally:
            if original_timm_module is None:
                del sys.modules["timm"]
            else:
                sys.modules["timm"] = original_timm_module
            if original_timm_models_module is None:
                del sys.modules["timm.models"]
            else:
                sys.modules["timm.models"] = original_timm_models_module
            if original_timm_layers_module is None:
                del sys.modules["timm.models.layers"]
            else:
                sys.modules["timm.models.layers"] = original_timm_layers_module
            if original_cv2_module is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = original_cv2_module

        class FakeAuxPreAlign(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shape: tuple[int, ...] | None = None

            def forward(self, y: torch.Tensor) -> torch.Tensor:
                self.input_shape = tuple(y.shape)
                return y.repeat(1, 3, 1, 1)

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.aux_shape: tuple[int, ...] | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                self.aux_shape = tuple(y.shape)
                batch_size = x.shape[0]
                return torch.ones(batch_size, 256, 2, 2), torch.ones(batch_size, 256, 2, 2)

        class FakeFusion(torch.nn.Module):
            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
                return rgb + aux

        class FakeDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                del res1, res2, res3, res4
                self.output_size = (h, w)
                return torch.zeros(2, 6, h, w)

        model = UNetFormerPreAlign.__new__(UNetFormerPreAlign)
        torch.nn.Module.__init__(model)
        model.aux_prealign = FakeAuxPreAlign()
        model.image_encoder = FakeImageEncoder()
        model.fpn1x = torch.nn.Identity()
        model.fpn2x = torch.nn.Identity()
        model.fpn3x = torch.nn.Identity()
        model.fpn4x = torch.nn.Identity()
        model.fpn1y = torch.nn.Identity()
        model.fpn2y = torch.nn.Identity()
        model.fpn3y = torch.nn.Identity()
        model.fpn4y = torch.nn.Identity()
        model.fusion1 = FakeFusion()
        model.fusion2 = FakeFusion()
        model.fusion3 = FakeFusion()
        model.fusion4 = FakeFusion()
        model.decoder = FakeDecoder()

        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8))

        self.assertEqual(model.aux_prealign.input_shape, (2, 1, 8, 8))
        self.assertEqual(model.image_encoder.aux_shape, (2, 3, 8, 8))
        self.assertEqual(model.decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_dga_repeats_auxiliary_input_and_applies_dga(self) -> None:
        fake_timm_module = types.ModuleType("timm")
        fake_timm_models_module = types.ModuleType("timm.models")
        fake_timm_layers_module = types.ModuleType("timm.models.layers")
        fake_timm_layers_module.DropPath = torch.nn.Identity
        fake_timm_layers_module.to_2tuple = lambda value: (value, value)
        fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
        fake_cv2_module = types.ModuleType("cv2")
        fake_cv2_module.COLORMAP_JET = 2
        fake_cv2_module.applyColorMap = lambda image, color_map: image
        fake_cv2_module.imwrite = lambda path, image: True
        original_timm_module = sys.modules.get("timm")
        original_timm_models_module = sys.modules.get("timm.models")
        original_timm_layers_module = sys.modules.get("timm.models.layers")
        original_cv2_module = sys.modules.get("cv2")

        try:
            sys.modules["timm"] = fake_timm_module
            sys.modules["timm.models"] = fake_timm_models_module
            sys.modules["timm.models.layers"] = fake_timm_layers_module
            sys.modules["cv2"] = fake_cv2_module
            from models.mfnet.UNetFormer_MMSAM_dga10 import UNetFormerDGA10
        finally:
            if original_timm_module is None:
                del sys.modules["timm"]
            else:
                sys.modules["timm"] = original_timm_module
            if original_timm_models_module is None:
                del sys.modules["timm.models"]
            else:
                sys.modules["timm.models"] = original_timm_models_module
            if original_timm_layers_module is None:
                del sys.modules["timm.models.layers"]
            else:
                sys.modules["timm.models.layers"] = original_timm_layers_module
            if original_cv2_module is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = original_cv2_module

        events: list[str] = []

        class FakePatchEmbed(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shapes: list[tuple[int, ...]] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.input_shapes.append(tuple(x.shape))
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.input_rgb: torch.Tensor | None = None
                self.input_aux: torch.Tensor | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.input_rgb = x.detach().clone()
                self.input_aux = y.detach().clone()
                return x + self.rgb_offset, y + self.aux_offset

        class FakeNeck(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: list[torch.Tensor] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append("neck")
                self.inputs.append(x.detach().clone())
                return x

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=14, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                        FakeBlock("block5", window_size=0, rgb_offset=6.0, aux_offset=60.0),
                    ]
                )
                self.neck = FakeNeck()

        class FakeFPN(torch.nn.Module):
            def __init__(self, name: str, offset: float) -> None:
                super().__init__()
                self.name = name
                self.offset = offset
                self.last_output: torch.Tensor | None = None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append(self.name)
                self.last_output = x + self.offset
                return self.last_output

        class SpyDGA(torch.nn.Module):
            def __init__(self, name: str, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.calls.append((rgb.detach().clone(), aux.detach().clone()))
                self.output_rgb = rgb.detach().clone() + self.rgb_offset
                self.output_aux = aux.detach().clone() + self.aux_offset
                return rgb + self.rgb_offset, aux + self.aux_offset

        class SpyFusion(torch.nn.Module):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name
                self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
                events.append(self.name)
                self.calls.append((rgb.detach().clone(), aux.detach().clone()))
                return rgb + aux

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                del res2, res3, res4
                events.append("decoder")
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        model = UNetFormerDGA10.__new__(UNetFormerDGA10)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()
        model.dga_indexes = [0, 2, 4, 5]
        model.dga_blocks = torch.nn.ModuleList(
            [
                SpyDGA("dga10_0", rgb_offset=100.0, aux_offset=1000.0),
                SpyDGA("dga10_1", rgb_offset=200.0, aux_offset=2000.0),
                SpyDGA("dga10_2", rgb_offset=300.0, aux_offset=3000.0),
                SpyDGA("dga10_3", rgb_offset=400.0, aux_offset=4000.0),
            ]
        )
        model.fpn1x = FakeFPN("fpn1x", 1.0)
        model.fpn2x = FakeFPN("fpn2x", 2.0)
        model.fpn3x = FakeFPN("fpn3x", 3.0)
        model.fpn4x = FakeFPN("fpn4x", 4.0)
        model.fpn1y = FakeFPN("fpn1y", 10.0)
        model.fpn2y = FakeFPN("fpn2y", 20.0)
        model.fpn3y = FakeFPN("fpn3y", 30.0)
        model.fpn4y = FakeFPN("fpn4y", 40.0)
        model.fusion1 = SpyFusion("fusion1")
        model.fusion2 = SpyFusion("fusion2")
        model.fusion3 = SpyFusion("fusion3")
        model.fusion4 = SpyFusion("fusion4")
        model.decoder = SpyDecoder()

        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8))

        self.assertFalse(hasattr(model, "aux_prealign"))
        self.assertEqual(model.image_encoder.patch_embed.input_shapes[1], (2, 3, 8, 8))
        self.assertEqual([len(dga_block.calls) for dga_block in model.dga_blocks], [1, 1, 1, 1])
        for dga_block in model.dga_blocks:
            self.assertEqual(dga_block.calls[0][0].shape, (2, 4, 2, 2))
            self.assertEqual(dga_block.calls[0][1].shape, (2, 4, 2, 2))

        expected_order = [
            "block0",
            "dga10_0",
            "block1",
            "block2",
            "dga10_1",
            "block3",
            "block4",
            "dga10_2",
            "block5",
            "dga10_3",
        ]
        self.assertEqual(events[: len(expected_order)], expected_order)
        self.assertLess(events.index("dga10_3"), events.index("neck"))
        self.assertEqual(model.decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_dga2_applies_dga2_after_global_blocks_with_bchw_boundary(self) -> None:
        fake_timm_module = types.ModuleType("timm")
        fake_timm_models_module = types.ModuleType("timm.models")
        fake_timm_layers_module = types.ModuleType("timm.models.layers")
        fake_timm_layers_module.DropPath = torch.nn.Identity
        fake_timm_layers_module.to_2tuple = lambda value: (value, value)
        fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
        fake_cv2_module = types.ModuleType("cv2")
        fake_cv2_module.COLORMAP_JET = 2
        fake_cv2_module.applyColorMap = lambda image, color_map: image
        fake_cv2_module.imwrite = lambda path, image: True
        original_timm_module = sys.modules.get("timm")
        original_timm_models_module = sys.modules.get("timm.models")
        original_timm_layers_module = sys.modules.get("timm.models.layers")
        original_cv2_module = sys.modules.get("cv2")

        try:
            sys.modules["timm"] = fake_timm_module
            sys.modules["timm.models"] = fake_timm_models_module
            sys.modules["timm.models.layers"] = fake_timm_layers_module
            sys.modules["cv2"] = fake_cv2_module
            from models.mfnet.UNetFormer_MMSAM_dga20 import UNetFormerDGA20
        finally:
            if original_timm_module is None:
                del sys.modules["timm"]
            else:
                sys.modules["timm"] = original_timm_module
            if original_timm_models_module is None:
                del sys.modules["timm.models"]
            else:
                sys.modules["timm.models"] = original_timm_models_module
            if original_timm_layers_module is None:
                del sys.modules["timm.models.layers"]
            else:
                sys.modules["timm.models.layers"] = original_timm_layers_module
            if original_cv2_module is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = original_cv2_module

        events: list[str] = []

        class FakePatchEmbed(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shapes: list[tuple[int, ...]] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.input_shapes.append(tuple(x.shape))
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.input_rgb: torch.Tensor | None = None
                self.input_aux: torch.Tensor | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.input_rgb = x.detach().clone()
                self.input_aux = y.detach().clone()
                return x + self.rgb_offset, y + self.aux_offset

        class FakeNeck(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: list[torch.Tensor] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append("neck")
                self.inputs.append(x.detach().clone())
                return x

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=14, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                        FakeBlock("block5", window_size=0, rgb_offset=6.0, aux_offset=60.0),
                    ]
                )
                self.neck = FakeNeck()

        class SpyDGA2(torch.nn.Module):
            def __init__(self, name: str, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None
                self.alpha = torch.nn.Parameter(torch.tensor([0.1]))
                self.beta = torch.nn.Parameter(torch.tensor([0.1]))

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.calls.append((rgb.detach().clone(), aux.detach().clone()))
                self.output_rgb = rgb.detach().clone() + self.rgb_offset
                self.output_aux = aux.detach().clone() + self.aux_offset
                return rgb + self.rgb_offset, aux + self.aux_offset

        class FakeFPN(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        class FakeFusion(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                del res2, res3, res4
                events.append("decoder")
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        model = UNetFormerDGA20.__new__(UNetFormerDGA20)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()
        model.dga_indexes = [0, 2, 4, 5]
        model.dga_blocks = torch.nn.ModuleList(
            [
                SpyDGA2("dga2_0", rgb_offset=100.0, aux_offset=1000.0),
                SpyDGA2("dga20", rgb_offset=200.0, aux_offset=2000.0),
                SpyDGA2("dga2_2", rgb_offset=300.0, aux_offset=3000.0),
                SpyDGA2("dga2_3", rgb_offset=400.0, aux_offset=4000.0),
            ]
        )
        model.fpn1x = FakeFPN()
        model.fpn2x = FakeFPN()
        model.fpn3x = FakeFPN()
        model.fpn4x = FakeFPN()
        model.fpn1y = FakeFPN()
        model.fpn2y = FakeFPN()
        model.fpn3y = FakeFPN()
        model.fpn4y = FakeFPN()
        model.fusion1 = FakeFusion()
        model.fusion2 = FakeFusion()
        model.fusion3 = FakeFusion()
        model.fusion4 = FakeFusion()
        model.decoder = SpyDecoder()

        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8))

        self.assertEqual(model.image_encoder.patch_embed.input_shapes[1], (2, 3, 8, 8))
        self.assertEqual([len(dga_block.calls) for dga_block in model.dga_blocks], [1, 1, 1, 1])
        for dga_block in model.dga_blocks:
            self.assertEqual(dga_block.calls[0][0].shape, (2, 4, 2, 2))
            self.assertEqual(dga_block.calls[0][1].shape, (2, 4, 2, 2))

        expected_order = [
            "block0",
            "dga2_0",
            "block1",
            "block2",
            "dga20",
            "block3",
            "block4",
            "dga2_2",
            "block5",
            "dga2_3",
        ]
        self.assertEqual(events[: len(expected_order)], expected_order)
        self.assertLess(events.index("dga2_3"), events.index("neck"))

        block1 = model.image_encoder.blocks[1]
        block3 = model.image_encoder.blocks[3]
        block5 = model.image_encoder.blocks[5]
        self.assertTrue(torch.equal(block1.input_rgb, model.dga_blocks[0].output_rgb.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block1.input_aux, model.dga_blocks[0].output_aux.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block3.input_rgb, model.dga_blocks[1].output_rgb.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block3.input_aux, model.dga_blocks[1].output_aux.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block5.input_rgb, model.dga_blocks[2].output_rgb.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block5.input_aux, model.dga_blocks[2].output_aux.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(model.image_encoder.neck.inputs[0], model.dga_blocks[3].output_rgb))
        self.assertTrue(torch.equal(model.image_encoder.neck.inputs[1], model.dga_blocks[3].output_aux))
        self.assertEqual(model.decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_dga20_dgsf10_uses_dga_features_and_encoder_final_top(self) -> None:
        fake_timm_module = types.ModuleType("timm")
        fake_timm_models_module = types.ModuleType("timm.models")
        fake_timm_layers_module = types.ModuleType("timm.models.layers")
        fake_timm_layers_module.DropPath = torch.nn.Identity
        fake_timm_layers_module.to_2tuple = lambda value: (value, value)
        fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
        fake_cv2_module = types.ModuleType("cv2")
        fake_cv2_module.COLORMAP_JET = 2
        fake_cv2_module.applyColorMap = lambda image, color_map: image
        fake_cv2_module.imwrite = lambda path, image: True
        original_timm_module = sys.modules.get("timm")
        original_timm_models_module = sys.modules.get("timm.models")
        original_timm_layers_module = sys.modules.get("timm.models.layers")
        original_cv2_module = sys.modules.get("cv2")

        try:
            sys.modules["timm"] = fake_timm_module
            sys.modules["timm.models"] = fake_timm_models_module
            sys.modules["timm.models.layers"] = fake_timm_layers_module
            sys.modules["cv2"] = fake_cv2_module
            from models.mfnet.UNetFormer_MMSAM_dga20_dgsf10 import UNetFormerDGA20DGSF10
        finally:
            if original_timm_module is None:
                del sys.modules["timm"]
            else:
                sys.modules["timm"] = original_timm_module
            if original_timm_models_module is None:
                del sys.modules["timm.models"]
            else:
                sys.modules["timm.models"] = original_timm_models_module
            if original_timm_layers_module is None:
                del sys.modules["timm.models.layers"]
            else:
                sys.modules["timm.models.layers"] = original_timm_layers_module
            if original_cv2_module is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = original_cv2_module

        events: list[str] = []

        class FakePatchEmbed(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.output_rgb = x.detach().clone() + self.rgb_offset
                self.output_aux = y.detach().clone() + self.aux_offset
                return x + self.rgb_offset, y + self.aux_offset

        class ForbiddenNeck(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                del x
                raise AssertionError("DGSF10 top features must be captured before encoder.neck")

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=0, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                        FakeBlock("block5", window_size=14, rgb_offset=6.0, aux_offset=60.0),
                    ]
                )
                self.neck = ForbiddenNeck()

        class SpyDGA(torch.nn.Module):
            def __init__(self, name: str, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.output_rgb = rgb.detach().clone() + self.rgb_offset
                self.output_aux = aux.detach().clone() + self.aux_offset
                return rgb + self.rgb_offset, aux + self.aux_offset

        class SpyDGSF10(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rgb_feats: tuple[torch.Tensor, ...] | None = None
                self.aux_feats: tuple[torch.Tensor, ...] | None = None
                self.outputs = tuple(torch.full((2, 4, 2, 2), float(index)) for index in range(1, 5))

            def forward(
                self,
                rgb_feats: tuple[torch.Tensor, ...],
                aux_feats: tuple[torch.Tensor, ...],
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                events.append("dgsf10")
                self.rgb_feats = tuple(feature.detach().clone() for feature in rgb_feats)
                self.aux_feats = tuple(feature.detach().clone() for feature in aux_feats)
                return self.outputs

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                events.append("decoder")
                self.inputs = (res1, res2, res3, res4)
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        model = UNetFormerDGA20DGSF10.__new__(UNetFormerDGA20DGSF10)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()
        model.dga_indexes = [0, 2, 3, 4]
        model.dga_blocks = torch.nn.ModuleList(
            [
                SpyDGA("dga0", rgb_offset=100.0, aux_offset=1000.0),
                SpyDGA("dga1", rgb_offset=200.0, aux_offset=2000.0),
                SpyDGA("dga2", rgb_offset=300.0, aux_offset=3000.0),
                SpyDGA("dga3", rgb_offset=400.0, aux_offset=4000.0),
            ]
        )
        model.dgsf10 = SpyDGSF10()
        model.decoder = SpyDecoder()

        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8))

        dgsf10 = model.dgsf10
        decoder = model.decoder
        assert isinstance(dgsf10, SpyDGSF10)
        assert isinstance(decoder, SpyDecoder)
        self.assertEqual(
            events,
            [
                "block0",
                "dga0",
                "block1",
                "block2",
                "dga1",
                "block3",
                "dga2",
                "block4",
                "dga3",
                "block5",
                "dgsf10",
                "decoder",
            ],
        )
        self.assertIsNotNone(dgsf10.rgb_feats)
        self.assertIsNotNone(dgsf10.aux_feats)
        self.assertEqual(len(dgsf10.rgb_feats), 5)
        self.assertEqual(len(dgsf10.aux_feats), 5)
        for index, dga_block in enumerate(model.dga_blocks):
            assert isinstance(dga_block, SpyDGA)
            self.assertTrue(torch.equal(dgsf10.rgb_feats[index], dga_block.output_rgb))
            self.assertTrue(torch.equal(dgsf10.aux_feats[index], dga_block.output_aux))
        final_block = model.image_encoder.blocks[-1]
        assert isinstance(final_block, FakeBlock)
        self.assertIsNotNone(final_block.output_rgb)
        self.assertIsNotNone(final_block.output_aux)
        self.assertTrue(torch.equal(dgsf10.rgb_feats[4], final_block.output_rgb.permute(0, 3, 1, 2)))
        self.assertTrue(torch.equal(dgsf10.aux_feats[4], final_block.output_aux.permute(0, 3, 1, 2)))
        self.assertIsNotNone(decoder.inputs)
        for actual, expected in zip(decoder.inputs, dgsf10.outputs):
            self.assertIs(actual, expected)
        self.assertEqual(decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_dgsf10_uses_encoder_features_and_encoder_final_top(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dgsf10 import UNetFormerDGSF10

        events: list[str] = []

        class FakePatchEmbed(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.output_rgb = x.detach().clone() + self.rgb_offset
                self.output_aux = y.detach().clone() + self.aux_offset
                return x + self.rgb_offset, y + self.aux_offset

        class ForbiddenNeck(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                del x
                raise AssertionError("DGSF10 top features must be captured before encoder.neck")

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=0, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                        FakeBlock("block5", window_size=14, rgb_offset=6.0, aux_offset=60.0),
                    ]
                )
                self.neck = ForbiddenNeck()

        class SpyDGSF10(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rgb_feats: tuple[torch.Tensor, ...] | None = None
                self.aux_feats: tuple[torch.Tensor, ...] | None = None
                self.outputs = tuple(torch.full((2, 4, 2, 2), float(index)) for index in range(1, 5))

            def forward(
                self,
                rgb_feats: tuple[torch.Tensor, ...],
                aux_feats: tuple[torch.Tensor, ...],
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                events.append("dgsf10")
                self.rgb_feats = tuple(feature.detach().clone() for feature in rgb_feats)
                self.aux_feats = tuple(feature.detach().clone() for feature in aux_feats)
                return self.outputs

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                events.append("decoder")
                self.inputs = (res1, res2, res3, res4)
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        model = UNetFormerDGSF10.__new__(UNetFormerDGSF10)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()
        model.dgsf10_indexes = [0, 2, 3, 4]
        model.dgsf10 = SpyDGSF10()
        model.decoder = SpyDecoder()

        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8))

        dgsf10 = model.dgsf10
        decoder = model.decoder
        assert isinstance(dgsf10, SpyDGSF10)
        assert isinstance(decoder, SpyDecoder)
        self.assertEqual(events, ["block0", "block1", "block2", "block3", "block4", "block5", "dgsf10", "decoder"])
        self.assertIsNotNone(dgsf10.rgb_feats)
        self.assertIsNotNone(dgsf10.aux_feats)
        self.assertEqual(len(dgsf10.rgb_feats), 5)
        self.assertEqual(len(dgsf10.aux_feats), 5)
        for feature_index, block_index in enumerate(model.dgsf10_indexes):
            block = model.image_encoder.blocks[block_index]
            assert isinstance(block, FakeBlock)
            self.assertIsNotNone(block.output_rgb)
            self.assertIsNotNone(block.output_aux)
            self.assertTrue(torch.equal(dgsf10.rgb_feats[feature_index], block.output_rgb.permute(0, 3, 1, 2)))
            self.assertTrue(torch.equal(dgsf10.aux_feats[feature_index], block.output_aux.permute(0, 3, 1, 2)))
        final_block = model.image_encoder.blocks[-1]
        assert isinstance(final_block, FakeBlock)
        self.assertIsNotNone(final_block.output_rgb)
        self.assertIsNotNone(final_block.output_aux)
        self.assertTrue(torch.equal(dgsf10.rgb_feats[4], final_block.output_rgb.permute(0, 3, 1, 2)))
        self.assertTrue(torch.equal(dgsf10.aux_feats[4], final_block.output_aux.permute(0, 3, 1, 2)))
        self.assertIsNotNone(decoder.inputs)
        for actual, expected in zip(decoder.inputs, dgsf10.outputs):
            self.assertIs(actual, expected)
        self.assertEqual(decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_prealign_auxalign_dgsf10_uses_aligned_aux_and_dgsf_features(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_prealign_auxalign_dgsf10 import (
                UNetFormerPreAlignAuxAlignDGSF10,
            )

        events: list[str] = []

        class FakeAuxPreAlign(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shape: tuple[int, ...] | None = None

            def forward(self, y: torch.Tensor) -> torch.Tensor:
                self.input_shape = tuple(y.shape)
                return y.repeat(1, 3, 1, 1)

        class FakePatchEmbed(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.output_rgb = x.detach().clone() + self.rgb_offset
                self.output_aux = y.detach().clone() + self.aux_offset
                return x + self.rgb_offset, y + self.aux_offset

        class ForbiddenNeck(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                del x
                raise AssertionError("DGSF10 features must be captured before encoder.neck")

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=0, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                        FakeBlock("block5", window_size=14, rgb_offset=6.0, aux_offset=60.0),
                    ]
                )
                self.neck = ForbiddenNeck()

        class SpyDGSF10(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rgb_feats: tuple[torch.Tensor, ...] | None = None
                self.aux_feats: tuple[torch.Tensor, ...] | None = None
                self.outputs = tuple(torch.full((2, 4, 2, 2), float(index)) for index in range(1, 5))

            def forward(
                self,
                rgb_feats: tuple[torch.Tensor, ...],
                aux_feats: tuple[torch.Tensor, ...],
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                events.append("dgsf10")
                self.rgb_feats = tuple(feature.detach().clone() for feature in rgb_feats)
                self.aux_feats = tuple(feature.detach().clone() for feature in aux_feats)
                return self.outputs

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                events.append("decoder")
                self.inputs = (res1, res2, res3, res4)
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        def make_model() -> torch.nn.Module:
            model = UNetFormerPreAlignAuxAlignDGSF10.__new__(UNetFormerPreAlignAuxAlignDGSF10)
            torch.nn.Module.__init__(model)
            model.aux_prealign = FakeAuxPreAlign()
            model.image_encoder = FakeImageEncoder()
            model.align_index = 2
            model.dgsf10_indexes = [0, 2, 3, 4]
            model.dgsf10 = SpyDGSF10()
            model.decoder = SpyDecoder()
            return model

        model = make_model()
        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8), return_align=False)

        dgsf10 = model.dgsf10
        decoder = model.decoder
        assert isinstance(dgsf10, SpyDGSF10)
        assert isinstance(decoder, SpyDecoder)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))
        self.assertEqual(model.aux_prealign.input_shape, (2, 1, 8, 8))
        self.assertEqual(events, ["block0", "block1", "block2", "block3", "block4", "block5", "dgsf10", "decoder"])
        self.assertIsNotNone(dgsf10.rgb_feats)
        self.assertIsNotNone(dgsf10.aux_feats)
        for feature_index, block_index in enumerate(model.dgsf10_indexes):
            block = model.image_encoder.blocks[block_index]
            assert isinstance(block, FakeBlock)
            self.assertTrue(torch.equal(dgsf10.rgb_feats[feature_index], block.output_rgb.permute(0, 3, 1, 2)))
            self.assertTrue(torch.equal(dgsf10.aux_feats[feature_index], block.output_aux.permute(0, 3, 1, 2)))
        final_block = model.image_encoder.blocks[-1]
        assert isinstance(final_block, FakeBlock)
        self.assertTrue(torch.equal(dgsf10.rgb_feats[4], final_block.output_rgb.permute(0, 3, 1, 2)))
        self.assertTrue(torch.equal(dgsf10.aux_feats[4], final_block.output_aux.permute(0, 3, 1, 2)))
        self.assertIsNotNone(decoder.inputs)
        for actual, expected in zip(decoder.inputs, dgsf10.outputs):
            self.assertIs(actual, expected)
        self.assertEqual(decoder.output_size, (8, 8))

        events.clear()
        model = make_model()
        logits, x_align_feat, y_align_feat = model(
            torch.zeros(2, 3, 8, 8),
            torch.ones(2, 8, 8),
            return_align=True,
        )

        self.assertEqual(tuple(logits.shape), (2, 6, 8, 8))
        self.assertEqual(x_align_feat.shape, y_align_feat.shape)
        align_block = model.image_encoder.blocks[2]
        assert isinstance(align_block, FakeBlock)
        self.assertTrue(torch.equal(x_align_feat, align_block.output_rgb))
        self.assertTrue(torch.equal(y_align_feat, align_block.output_aux))

    def test_attach_intermediate_stats_sets_recorder_and_prefix(self) -> None:
        from models.mfnet.intermediate_stats_config import attach_intermediate_stats

        owner = torch.nn.Module()
        owner.intermediate_stats = IntermediateStatsRecorder()
        module = torch.nn.Module()

        attach_intermediate_stats(owner, module, "dgsf10")

        self.assertIs(module.intermediate_stats, owner.intermediate_stats)
        self.assertEqual(module.intermediate_stats_prefix, "dgsf10")

    def test_attach_requested_intermediate_stats_mounts_requested_modules_only(self) -> None:
        from models.mfnet.intermediate_stats_config import attach_requested_intermediate_stats

        owner = torch.nn.Module()
        dga_block_0 = torch.nn.Module()
        dga_block_1 = torch.nn.Module()
        dgsf10 = torch.nn.Module()

        attach_requested_intermediate_stats(
            owner,
            ["unknown", "dga"],
            {
                "dga": [
                    (dga_block_0, "dga/block_0"),
                    (dga_block_1, "dga/block_1"),
                ],
                "dgsf10": [(dgsf10, "dgsf10")],
            },
        )

        self.assertIs(dga_block_0.intermediate_stats, owner.intermediate_stats)
        self.assertEqual(dga_block_0.intermediate_stats_prefix, "dga/block_0")
        self.assertIs(dga_block_1.intermediate_stats, owner.intermediate_stats)
        self.assertEqual(dga_block_1.intermediate_stats_prefix, "dga/block_1")
        self.assertFalse(hasattr(dgsf10, "intermediate_stats"))

        empty_owner = torch.nn.Module()
        unavailable_module = torch.nn.Module()
        attach_requested_intermediate_stats(
            empty_owner,
            ["unknown"],
            {"dga": [(unavailable_module, "dga/block_0")]},
        )

        self.assertFalse(hasattr(empty_owner, "intermediate_stats"))
        self.assertFalse(hasattr(unavailable_module, "intermediate_stats"))

    def test_unetformer_dgsf10_records_dgsf10_intermediate_stats_when_requested(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dgsf10 import UNetFormerDGSF10

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dgsf10.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGSF10(
                    record_intermediate_stats=True,
                    record_intermediate_modules=["dgsf10"],
                )

        self.assertIsNotNone(model.intermediate_stats)
        self.assertIs(model.dgsf10.intermediate_stats, model.intermediate_stats)

    def test_unetformer_dgsf10_records_no_intermediate_modules_by_default(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dgsf10 import UNetFormerDGSF10

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dgsf10.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGSF10(record_intermediate_stats=True)

        self.assertFalse(hasattr(model, "intermediate_stats"))
        self.assertFalse(hasattr(model.dgsf10, "intermediate_stats"))

    def test_unetformer_dgsf10_ignores_unavailable_intermediate_stats_modules(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dgsf10 import UNetFormerDGSF10

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dgsf10.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGSF10(
                    record_intermediate_stats=True,
                    record_intermediate_modules=["dga"],
                )

        self.assertFalse(hasattr(model, "intermediate_stats"))
        self.assertFalse(hasattr(model.dgsf10, "intermediate_stats"))

    def test_unetformer_prealign_auxalign_dgsf10_records_dgsf10_stats_when_requested(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_prealign_auxalign_dgsf10 import (
                UNetFormerPreAlignAuxAlignDGSF10,
            )

            def fake_parent_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )
                self.align_index = 0

            with patch(
                "models.mfnet.UNetFormer_MMSAM_prealign_auxalign_dgsf10.UNetFormerPreAlignAuxAlign.__init__",
                new=fake_parent_init,
            ):
                model = UNetFormerPreAlignAuxAlignDGSF10(
                    record_intermediate_stats=True,
                    record_intermediate_modules=["dgsf10"],
                )

        self.assertIsNotNone(model.intermediate_stats)
        self.assertIs(model.dgsf10.intermediate_stats, model.intermediate_stats)
        self.assertEqual(model.dgsf10.intermediate_stats_prefix, "dgsf10")

    def test_unetformer_prealign_auxalign_dgsf10_records_no_stats_by_default(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_prealign_auxalign_dgsf10 import (
                UNetFormerPreAlignAuxAlignDGSF10,
            )

            def fake_parent_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )
                self.align_index = 0

            with patch(
                "models.mfnet.UNetFormer_MMSAM_prealign_auxalign_dgsf10.UNetFormerPreAlignAuxAlign.__init__",
                new=fake_parent_init,
            ):
                model = UNetFormerPreAlignAuxAlignDGSF10(record_intermediate_stats=True)

        self.assertFalse(hasattr(model, "intermediate_stats"))
        self.assertFalse(hasattr(model.dgsf10, "intermediate_stats"))

    def test_unetformer_dga20_dgsf10_records_no_intermediate_modules_by_default(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dga20_dgsf10 import UNetFormerDGA20DGSF10

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dga20.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGA20DGSF10(record_intermediate_stats=True)

        self.assertFalse(hasattr(model, "intermediate_stats"))
        for block in model.dga_blocks:
            self.assertFalse(hasattr(block, "intermediate_stats"))
        self.assertFalse(hasattr(model.dgsf10, "intermediate_stats"))

    def test_unetformer_dga20_dgsf10_initializes_parent_without_intermediate_stats(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dga20_dgsf10 import UNetFormerDGA20DGSF10

            captured_kwargs: list[dict[str, object]] = []

            def fake_dga20_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args
                captured_kwargs.append(kwargs)
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(embed_dim=8)
                self.dga_blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])

            with patch("models.mfnet.UNetFormer_MMSAM_dga20_dgsf10.UNetFormerDGA20.__init__", new=fake_dga20_init):
                model = UNetFormerDGA20DGSF10(
                    record_intermediate_stats=True,
                    record_intermediate_modules=["dga", "dgsf10"],
                )

        self.assertEqual(captured_kwargs[0]["record_intermediate_stats"], False)
        self.assertEqual(captured_kwargs[0]["record_intermediate_modules"], ())
        for block in model.dga_blocks:
            self.assertIs(block.intermediate_stats, model.intermediate_stats)
        self.assertIs(model.dgsf10.intermediate_stats, model.intermediate_stats)

    def test_unetformer_dga20_dgsf10_can_record_only_dga_intermediate_stats(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dga20_dgsf10 import UNetFormerDGA20DGSF10

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dga20.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGA20DGSF10(
                    record_intermediate_stats=True,
                    record_intermediate_modules=["dga"],
                )

        for block in model.dga_blocks:
            self.assertIs(block.intermediate_stats, model.intermediate_stats)
        self.assertFalse(hasattr(model.dgsf10, "intermediate_stats"))

    def test_unetformer_dga20_dgsf10_can_record_only_dgsf10_intermediate_stats(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dga20_dgsf10 import UNetFormerDGA20DGSF10

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dga20.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGA20DGSF10(
                    record_intermediate_stats=True,
                    record_intermediate_modules=["unknown", "dgsf10"],
                )

        for block in model.dga_blocks:
            self.assertFalse(hasattr(block, "intermediate_stats"))
        self.assertIs(model.dgsf10.intermediate_stats, model.intermediate_stats)

    def test_unetformer_dga20_ignores_unavailable_intermediate_stats_modules(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dga20 import UNetFormerDGA20

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dga20.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGA20(
                    record_intermediate_stats=True,
                    record_intermediate_modules=["dgsf10"],
                )

        self.assertFalse(hasattr(model, "intermediate_stats"))
        for block in model.dga_blocks:
            self.assertFalse(hasattr(block, "intermediate_stats"))

    def test_record_intermediate_stats_false_disables_requested_modules(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dga20_dgsf10 import UNetFormerDGA20DGSF10

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dga20.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGA20DGSF10(
                    record_intermediate_stats=False,
                    record_intermediate_modules=["dga", "dgsf10"],
                )

        self.assertFalse(hasattr(model, "intermediate_stats"))
        for block in model.dga_blocks:
            self.assertFalse(hasattr(block, "intermediate_stats"))
        self.assertFalse(hasattr(model.dgsf10, "intermediate_stats"))

    def test_unetformer_prealign_dga_applies_dga_after_all_global_blocks(self) -> None:
        fake_timm_module = types.ModuleType("timm")
        fake_timm_models_module = types.ModuleType("timm.models")
        fake_timm_layers_module = types.ModuleType("timm.models.layers")
        fake_timm_layers_module.DropPath = torch.nn.Identity
        fake_timm_layers_module.to_2tuple = lambda value: (value, value)
        fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
        fake_cv2_module = types.ModuleType("cv2")
        fake_cv2_module.COLORMAP_JET = 2
        fake_cv2_module.applyColorMap = lambda image, color_map: image
        fake_cv2_module.imwrite = lambda path, image: True
        original_timm_module = sys.modules.get("timm")
        original_timm_models_module = sys.modules.get("timm.models")
        original_timm_layers_module = sys.modules.get("timm.models.layers")
        original_cv2_module = sys.modules.get("cv2")

        try:
            sys.modules["timm"] = fake_timm_module
            sys.modules["timm.models"] = fake_timm_models_module
            sys.modules["timm.models.layers"] = fake_timm_layers_module
            sys.modules["cv2"] = fake_cv2_module
            from models.mfnet.UNetFormer_MMSAM_prealign_dga10 import UNetFormerPreAlignDGA10
        finally:
            if original_timm_module is None:
                del sys.modules["timm"]
            else:
                sys.modules["timm"] = original_timm_module
            if original_timm_models_module is None:
                del sys.modules["timm.models"]
            else:
                sys.modules["timm.models"] = original_timm_models_module
            if original_timm_layers_module is None:
                del sys.modules["timm.models.layers"]
            else:
                sys.modules["timm.models.layers"] = original_timm_layers_module
            if original_cv2_module is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = original_cv2_module

        events: list[str] = []

        class FakeAuxPreAlign(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shape: tuple[int, ...] | None = None

            def forward(self, y: torch.Tensor) -> torch.Tensor:
                self.input_shape = tuple(y.shape)
                return y.repeat(1, 3, 1, 1)

        class FakePatchEmbed(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.input_rgb: torch.Tensor | None = None
                self.input_aux: torch.Tensor | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.input_rgb = x.detach().clone()
                self.input_aux = y.detach().clone()
                return x + self.rgb_offset, y + self.aux_offset

        class FakeNeck(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: list[torch.Tensor] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append("neck")
                self.inputs.append(x.detach().clone())
                return x

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=14, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                        FakeBlock("block5", window_size=0, rgb_offset=6.0, aux_offset=60.0),
                    ]
                )
                self.neck = FakeNeck()

        class FakeFPN(torch.nn.Module):
            def __init__(self, name: str, offset: float) -> None:
                super().__init__()
                self.name = name
                self.offset = offset
                self.last_output: torch.Tensor | None = None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append(self.name)
                self.last_output = x + self.offset
                return self.last_output

        class SpyDGA(torch.nn.Module):
            def __init__(self, name: str, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.calls.append((rgb.detach().clone(), aux.detach().clone()))
                self.output_rgb = rgb.detach().clone() + self.rgb_offset
                self.output_aux = aux.detach().clone() + self.aux_offset
                return rgb + self.rgb_offset, aux + self.aux_offset

        class SpyFusion(torch.nn.Module):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name
                self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
                events.append(self.name)
                self.calls.append((rgb.detach().clone(), aux.detach().clone()))
                return rgb + aux

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                events.append("decoder")
                self.inputs = (
                    res1.detach().clone(),
                    res2.detach().clone(),
                    res3.detach().clone(),
                    res4.detach().clone(),
                )
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        model = UNetFormerPreAlignDGA10.__new__(UNetFormerPreAlignDGA10)
        torch.nn.Module.__init__(model)
        model.aux_prealign = FakeAuxPreAlign()
        model.image_encoder = FakeImageEncoder()
        model.dga_indexes = [0, 2, 4, 5]
        model.dga_blocks = torch.nn.ModuleList(
            [
                SpyDGA("dga10_0", rgb_offset=100.0, aux_offset=1000.0),
                SpyDGA("dga10_1", rgb_offset=200.0, aux_offset=2000.0),
                SpyDGA("dga10_2", rgb_offset=300.0, aux_offset=3000.0),
                SpyDGA("dga10_3", rgb_offset=400.0, aux_offset=4000.0),
            ]
        )
        model.fpn1x = FakeFPN("fpn1x", 1.0)
        model.fpn2x = FakeFPN("fpn2x", 2.0)
        model.fpn3x = FakeFPN("fpn3x", 3.0)
        model.fpn4x = FakeFPN("fpn4x", 4.0)
        model.fpn1y = FakeFPN("fpn1y", 10.0)
        model.fpn2y = FakeFPN("fpn2y", 20.0)
        model.fpn3y = FakeFPN("fpn3y", 30.0)
        model.fpn4y = FakeFPN("fpn4y", 40.0)
        model.fusion1 = SpyFusion("fusion1")
        model.fusion2 = SpyFusion("fusion2")
        model.fusion3 = SpyFusion("fusion3")
        model.fusion4 = SpyFusion("fusion4")
        model.decoder = SpyDecoder()

        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8))

        self.assertEqual(model.aux_prealign.input_shape, (2, 1, 8, 8))
        self.assertEqual([len(dga_block.calls) for dga_block in model.dga_blocks], [1, 1, 1, 1])
        for dga_block in model.dga_blocks:
            self.assertEqual(dga_block.calls[0][0].shape, (2, 4, 2, 2))
            self.assertEqual(dga_block.calls[0][1].shape, (2, 4, 2, 2))

        expected_order = [
            "block0",
            "dga10_0",
            "block1",
            "block2",
            "dga10_1",
            "block3",
            "block4",
            "dga10_2",
            "block5",
            "dga10_3",
        ]
        self.assertEqual(events[: len(expected_order)], expected_order)
        self.assertLess(events.index("dga10_3"), events.index("neck"))

        block1 = model.image_encoder.blocks[1]
        block3 = model.image_encoder.blocks[3]
        block5 = model.image_encoder.blocks[5]
        self.assertTrue(torch.equal(block1.input_rgb, model.dga_blocks[0].output_rgb.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block1.input_aux, model.dga_blocks[0].output_aux.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block3.input_rgb, model.dga_blocks[1].output_rgb.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block3.input_aux, model.dga_blocks[1].output_aux.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block5.input_rgb, model.dga_blocks[2].output_rgb.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block5.input_aux, model.dga_blocks[2].output_aux.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(model.image_encoder.neck.inputs[0], model.dga_blocks[3].output_rgb))
        self.assertTrue(torch.equal(model.image_encoder.neck.inputs[1], model.dga_blocks[3].output_aux))

        self.assertLess(events.index("fusion1"), events.index("decoder"))
        self.assertLess(events.index("fusion2"), events.index("decoder"))
        self.assertLess(events.index("fusion3"), events.index("decoder"))
        self.assertLess(events.index("fusion4"), events.index("decoder"))
        self.assertIsNotNone(model.fpn1x.last_output)
        self.assertIsNotNone(model.fpn1y.last_output)
        self.assertTrue(torch.equal(model.fusion1.calls[0][0], model.fpn1x.last_output))
        self.assertTrue(torch.equal(model.fusion1.calls[0][1], model.fpn1y.last_output))
        self.assertEqual(model.decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_prealign_dga_requires_four_global_blocks(self) -> None:
        fake_timm_module = types.ModuleType("timm")
        fake_timm_models_module = types.ModuleType("timm.models")
        fake_timm_layers_module = types.ModuleType("timm.models.layers")
        fake_timm_layers_module.DropPath = torch.nn.Identity
        fake_timm_layers_module.to_2tuple = lambda value: (value, value)
        fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
        fake_cv2_module = types.ModuleType("cv2")
        fake_cv2_module.COLORMAP_JET = 2
        fake_cv2_module.applyColorMap = lambda image, color_map: image
        fake_cv2_module.imwrite = lambda path, image: True
        original_timm_module = sys.modules.get("timm")
        original_timm_models_module = sys.modules.get("timm.models")
        original_timm_layers_module = sys.modules.get("timm.models.layers")
        original_cv2_module = sys.modules.get("cv2")

        try:
            sys.modules["timm"] = fake_timm_module
            sys.modules["timm.models"] = fake_timm_models_module
            sys.modules["timm.models.layers"] = fake_timm_layers_module
            sys.modules["cv2"] = fake_cv2_module
            from models.mfnet.UNetFormer_MMSAM_prealign_dga10 import UNetFormerPreAlignDGA10
        finally:
            if original_timm_module is None:
                del sys.modules["timm"]
            else:
                sys.modules["timm"] = original_timm_module
            if original_timm_models_module is None:
                del sys.modules["timm.models"]
            else:
                sys.modules["timm.models"] = original_timm_models_module
            if original_timm_layers_module is None:
                del sys.modules["timm.models.layers"]
            else:
                sys.modules["timm.models.layers"] = original_timm_layers_module
            if original_cv2_module is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = original_cv2_module

        class FakeBlock(torch.nn.Module):
            def __init__(self, window_size: int) -> None:
                super().__init__()
                self.window_size = window_size

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock(window_size=0),
                        FakeBlock(window_size=14),
                        FakeBlock(window_size=0),
                    ]
                )

        model = UNetFormerPreAlignDGA10.__new__(UNetFormerPreAlignDGA10)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()

        with self.assertRaises(ValueError):
            model._resolve_dga_indexes()

    def test_unetformer_prealign_auxalign_dga_applies_dga_and_returns_align_features(self) -> None:
        fake_timm_module = types.ModuleType("timm")
        fake_timm_models_module = types.ModuleType("timm.models")
        fake_timm_layers_module = types.ModuleType("timm.models.layers")
        fake_timm_layers_module.DropPath = torch.nn.Identity
        fake_timm_layers_module.to_2tuple = lambda value: (value, value)
        fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
        fake_cv2_module = types.ModuleType("cv2")
        fake_cv2_module.COLORMAP_JET = 2
        fake_cv2_module.applyColorMap = lambda image, color_map: image
        fake_cv2_module.imwrite = lambda path, image: True
        original_timm_module = sys.modules.get("timm")
        original_timm_models_module = sys.modules.get("timm.models")
        original_timm_layers_module = sys.modules.get("timm.models.layers")
        original_cv2_module = sys.modules.get("cv2")

        try:
            sys.modules["timm"] = fake_timm_module
            sys.modules["timm.models"] = fake_timm_models_module
            sys.modules["timm.models.layers"] = fake_timm_layers_module
            sys.modules["cv2"] = fake_cv2_module
            from models.mfnet.UNetFormer_MMSAM_prealign_auxalign_dga10 import UNetFormerPreAlignAuxAlignDGA10
        finally:
            if original_timm_module is None:
                del sys.modules["timm"]
            else:
                sys.modules["timm"] = original_timm_module
            if original_timm_models_module is None:
                del sys.modules["timm.models"]
            else:
                sys.modules["timm.models"] = original_timm_models_module
            if original_timm_layers_module is None:
                del sys.modules["timm.models.layers"]
            else:
                sys.modules["timm.models.layers"] = original_timm_layers_module
            if original_cv2_module is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = original_cv2_module

        events: list[str] = []

        class FakeAuxPreAlign(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shape: tuple[int, ...] | None = None

            def forward(self, y: torch.Tensor) -> torch.Tensor:
                self.input_shape = tuple(y.shape)
                return y.repeat(1, 3, 1, 1)

        class FakePatchEmbed(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                return x + self.rgb_offset, y + self.aux_offset

        class FakeNeck(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: list[torch.Tensor] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append("neck")
                self.inputs.append(x.detach().clone())
                return x

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=0, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                    ]
                )
                self.neck = FakeNeck()

        class SpyDGA(torch.nn.Module):
            def __init__(self, name: str, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.calls.append((rgb.detach().clone(), aux.detach().clone()))
                self.output_rgb = rgb.detach().clone() + self.rgb_offset
                self.output_aux = aux.detach().clone() + self.aux_offset
                return rgb + self.rgb_offset, aux + self.aux_offset

        class SpyFusion(torch.nn.Module):
            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
                return rgb + aux

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output_size: tuple[int, int] | None = None
                self.inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                events.append("decoder")
                self.inputs = (res1.detach().clone(), res2.detach().clone(), res3.detach().clone(), res4.detach().clone())
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        def make_model() -> torch.nn.Module:
            model = UNetFormerPreAlignAuxAlignDGA10.__new__(UNetFormerPreAlignAuxAlignDGA10)
            torch.nn.Module.__init__(model)
            model.aux_prealign = FakeAuxPreAlign()
            model.image_encoder = FakeImageEncoder()
            model.align_index = 2
            model.dga_indexes = [0, 2, 3, 4]
            model.dga_blocks = torch.nn.ModuleList(
                [
                    SpyDGA("dga10_0", rgb_offset=100.0, aux_offset=1000.0),
                    SpyDGA("dga10_1", rgb_offset=200.0, aux_offset=2000.0),
                    SpyDGA("dga10_2", rgb_offset=300.0, aux_offset=3000.0),
                    SpyDGA("dga10_3", rgb_offset=400.0, aux_offset=4000.0),
                ]
            )
            model.fpn1x = torch.nn.Identity()
            model.fpn2x = torch.nn.Identity()
            model.fpn3x = torch.nn.Identity()
            model.fpn4x = torch.nn.Identity()
            model.fpn1y = torch.nn.Identity()
            model.fpn2y = torch.nn.Identity()
            model.fpn3y = torch.nn.Identity()
            model.fpn4y = torch.nn.Identity()
            model.fusion1 = SpyFusion()
            model.fusion2 = SpyFusion()
            model.fusion3 = SpyFusion()
            model.fusion4 = SpyFusion()
            model.decoder = SpyDecoder()
            return model

        model = make_model()
        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8), return_align=False)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))
        self.assertEqual(model.aux_prealign.input_shape, (2, 1, 8, 8))
        self.assertEqual([len(dga_block.calls) for dga_block in model.dga_blocks], [1, 1, 1, 1])
        self.assertEqual(
            events[:9],
            ["block0", "dga10_0", "block1", "block2", "dga10_1", "block3", "dga10_2", "block4", "dga10_3"],
        )
        self.assertLess(events.index("dga10_3"), events.index("neck"))
        self.assertTrue(torch.equal(model.image_encoder.neck.inputs[0], model.dga_blocks[3].output_rgb))
        self.assertTrue(torch.equal(model.image_encoder.neck.inputs[1], model.dga_blocks[3].output_aux))
        self.assertEqual(model.decoder.output_size, (8, 8))

        events.clear()
        model = make_model()
        logits, x_align_feat, y_align_feat = model(
            torch.zeros(2, 3, 8, 8),
            torch.ones(2, 8, 8),
            return_align=True,
        )

        self.assertEqual(tuple(logits.shape), (2, 6, 8, 8))
        self.assertEqual(x_align_feat.shape, y_align_feat.shape)
        self.assertTrue(torch.equal(x_align_feat, model.dga_blocks[1].calls[0][0].permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(y_align_feat, model.dga_blocks[1].calls[0][1].permute(0, 2, 3, 1)))

    def test_unetformer_dga_requires_four_global_blocks(self) -> None:
        fake_timm_module = types.ModuleType("timm")
        fake_timm_models_module = types.ModuleType("timm.models")
        fake_timm_layers_module = types.ModuleType("timm.models.layers")
        fake_timm_layers_module.DropPath = torch.nn.Identity
        fake_timm_layers_module.to_2tuple = lambda value: (value, value)
        fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
        fake_cv2_module = types.ModuleType("cv2")
        fake_cv2_module.COLORMAP_JET = 2
        fake_cv2_module.applyColorMap = lambda image, color_map: image
        fake_cv2_module.imwrite = lambda path, image: True
        original_timm_module = sys.modules.get("timm")
        original_timm_models_module = sys.modules.get("timm.models")
        original_timm_layers_module = sys.modules.get("timm.models.layers")
        original_cv2_module = sys.modules.get("cv2")

        try:
            sys.modules["timm"] = fake_timm_module
            sys.modules["timm.models"] = fake_timm_models_module
            sys.modules["timm.models.layers"] = fake_timm_layers_module
            sys.modules["cv2"] = fake_cv2_module
            from models.mfnet.UNetFormer_MMSAM_dga10 import UNetFormerDGA10
        finally:
            if original_timm_module is None:
                del sys.modules["timm"]
            else:
                sys.modules["timm"] = original_timm_module
            if original_timm_models_module is None:
                del sys.modules["timm.models"]
            else:
                sys.modules["timm.models"] = original_timm_models_module
            if original_timm_layers_module is None:
                del sys.modules["timm.models.layers"]
            else:
                sys.modules["timm.models.layers"] = original_timm_layers_module
            if original_cv2_module is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = original_cv2_module

        class FakeBlock(torch.nn.Module):
            def __init__(self, window_size: int) -> None:
                super().__init__()
                self.window_size = window_size

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock(window_size=0),
                        FakeBlock(window_size=14),
                        FakeBlock(window_size=0),
                    ]
                )

        model = UNetFormerDGA10.__new__(UNetFormerDGA10)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()

        with self.assertRaises(ValueError):
            model._resolve_dga_indexes()

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
            "seed": 80,
            "model": {
                "type": "mfnet_unetformer",
                "num_classes": 6,
                "sam_backbone": "vit_b",
                "sam_checkpoint": "/tmp/sam_vit_b.pth",
            },
            "dataset": {
                "name": "vaihingen",
                "root_dir": root_dir,
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
        cfg: dict[str, object] | None,
        default_work_dir: Path | None,
    ) -> dict[str, object]:
        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.image_encoder = torch.nn.Linear(1, 1, bias=False)
                self.align_index = 2

        captured_dataset_calls: list[dict[str, object]] = []
        captured_trainer_kwargs: list[dict[str, object]] = []
        captured_trainer_classes: list[str] = []
        captured_model_cfg: list[dict[str, object]] = []
        captured_load_config_paths: list[str] = []
        captured_default_work_dir_calls: list[dict[str, object]] = []

        class FakeTrainer:
            def __init__(self, **kwargs: object) -> None:
                captured_trainer_classes.append("MFNetTrainer")
                captured_trainer_kwargs.append(kwargs)

            def train(self) -> None:
                return None

        class FakeDGATrainer:
            def __init__(self, **kwargs: object) -> None:
                captured_trainer_classes.append("MFNetDGATrainer")
                captured_trainer_kwargs.append(kwargs)

            def train(self) -> None:
                return None

        class FakeAuxAlignTrainer:
            def __init__(self, **kwargs: object) -> None:
                captured_trainer_classes.append("MFNetAuxAlignTrainer")
                captured_trainer_kwargs.append(kwargs)

            def train(self) -> None:
                return None

        class FakeAuxAlignDGATrainer:
            def __init__(self, **kwargs: object) -> None:
                captured_trainer_classes.append("MFNetAuxAlignDGATrainer")
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
        original_dga_trainer = module.MFNetDGATrainer
        original_auxalign_trainer = module.MFNetAuxAlignTrainer
        original_auxalign_dga_trainer = module.MFNetAuxAlignDGATrainer
        try:
            module.parse_args = lambda: args

            def fake_load_config(path: str) -> dict[str, object]:
                captured_load_config_paths.append(path)
                if cfg is None:
                    return original_load_config(path)
                return cfg

            module.load_config = fake_load_config

            def fake_build_default_work_dir(
                model_name: object,
                dataset_name: object,
                seed: object,
                lambda_align: object | None = None,
                root_dir: str | Path = "work_dirs",
            ) -> Path:
                captured_default_work_dir_calls.append(
                    {
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "seed": seed,
                        "lambda_align": lambda_align,
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
            module.MFNetDGATrainer = FakeDGATrainer
            module.MFNetAuxAlignTrainer = FakeAuxAlignTrainer
            module.MFNetAuxAlignDGATrainer = FakeAuxAlignDGATrainer

            module.main()
        finally:
            module.parse_args = original_parse_args
            module.load_config = original_load_config
            module.build_default_work_dir = original_build_default_work_dir
            module.build_model = original_build_model
            module.build_isprs_dataset = original_build_isprs_dataset
            module.DataLoader = original_dataloader
            module.MFNetTrainer = original_trainer
            module.MFNetDGATrainer = original_dga_trainer
            module.MFNetAuxAlignTrainer = original_auxalign_trainer
            module.MFNetAuxAlignDGATrainer = original_auxalign_dga_trainer

        return {
            "dataset_calls": captured_dataset_calls,
            "trainer_kwargs": captured_trainer_kwargs,
            "trainer_classes": captured_trainer_classes,
            "model_cfg": captured_model_cfg,
            "load_config_paths": captured_load_config_paths,
            "default_work_dir_calls": captured_default_work_dir_calls,
        }

    def test_train_default_work_dir_uses_model_dataset_and_short_run_id(self) -> None:
        module = self._load_train_entry_module()

        with patch.object(sys, "argv", ["train.py"]):
            args = module.parse_args()

        self.assertFalse(hasattr(args, "work_dir"))
        self.assertFalse(hasattr(args, "resume_from"))
        self.assertIsNone(args.resume_dir)
        self.assertIsNone(args.resume_ckpt)
        self.assertIsNone(args.model_type)
        self.assertIsNone(args.seed)

        with patch.object(sys, "argv", ["train.py", "--model-type", "mfnet_unetformer_dga10", "--seed", "123"]):
            args = module.parse_args()

        self.assertEqual(args.model_type, "mfnet_unetformer_dga10")
        self.assertEqual(args.seed, 123)

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = module.build_default_work_dir(
                model_name="MFNet UNetFormer",
                dataset_name="Potsdam/RGB",
                seed=123,
                root_dir=tmpdir,
            )

            self.assertEqual(work_dir.parent, Path(tmpdir))
            self.assertRegex(
                work_dir.name,
                re.compile(r"^potsdam_rgb_base_123_[0-9a-f]{5}$"),
            )

            dga_work_dir = module.build_default_work_dir(
                model_name="mfnet_unetformer_dga20",
                dataset_name="vaihingen",
                seed=80,
                root_dir=tmpdir,
            )
            self.assertRegex(
                dga_work_dir.name,
                re.compile(r"^vaihingen_dga20_80_[0-9a-f]{5}$"),
            )

            lambda_work_dir = module.build_default_work_dir(
                model_name="mfnet_unetformer_prealign_auxalign",
                dataset_name="vaihingen",
                seed=80,
                lambda_align=0.01,
                root_dir=tmpdir,
            )
            self.assertRegex(
                lambda_work_dir.name,
                re.compile(r"^vaihingen_prealign_auxalign_80_lambda-0.01_[0-9a-f]{5}$"),
            )

    def test_optimizer_param_groups_exclude_gate_scalars_from_weight_decay(self) -> None:
        module = self._load_train_entry_module()

        class FakeGateModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.alpha = torch.nn.Parameter(torch.ones(1))
                self.beta = torch.nn.Parameter(torch.ones(1))
                self.gamma = torch.nn.Parameter(torch.ones(1))
                self.lora_alpha = torch.nn.Parameter(torch.ones(1))
                self.register_parameter("lambda", torch.nn.Parameter(torch.ones(1)))

        model = FakeGateModel()
        param_groups = module.build_optimizer_param_groups(model, weight_decay=0.0005)

        self.assertEqual(len(param_groups), 2)
        decay_group = next(group for group in param_groups if group["weight_decay"] == 0.0005)
        no_decay_group = next(group for group in param_groups if group["weight_decay"] == 0.0)
        decay_param_ids = {id(param) for param in decay_group["params"]}
        no_decay_param_ids = {id(param) for param in no_decay_group["params"]}

        self.assertIn(id(model.weight), decay_param_ids)
        self.assertIn(id(model.lora_alpha), decay_param_ids)
        self.assertIn(id(model.alpha), no_decay_param_ids)
        self.assertIn(id(model.beta), no_decay_param_ids)
        self.assertIn(id(model.gamma), no_decay_param_ids)
        self.assertIn(id(model._parameters["lambda"]), no_decay_param_ids)
        self.assertFalse(decay_param_ids & no_decay_param_ids)

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
            self.assertEqual(dataset_calls[0]["ids"], ["1"])
            self.assertEqual(dataset_calls[1]["ids"], ["5"])
            self.assertEqual(result["load_config_paths"], [str(config_path)])
            self.assertEqual(len(result["default_work_dir_calls"]), 1)
            self.assertEqual(
                result["default_work_dir_calls"][0],
                {
                    "model_name": "mfnet_unetformer",
                    "dataset_name": "vaihingen",
                    "seed": 80,
                    "lambda_align": None,
                    "root_dir": "work_dirs",
                },
            )
            self.assertEqual(len(result["trainer_kwargs"]), 1)
            self.assertEqual(result["trainer_classes"], ["MFNetTrainer"])
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["optimizer"], torch.optim.SGD)
            self.assertIsInstance(
                trainer_kwargs["scheduler"],
                torch.optim.lr_scheduler.MultiStepLR,
            )
            self.assertEqual(len(result["model_cfg"]), 1)
            self.assertEqual(result["model_cfg"][0]["sam_checkpoint"], "/tmp/sam_vit_b.pth")
            self.assertIsInstance(trainer_kwargs["logger"], TestNetLogger)
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
            saved_cfg = json.loads((work_dir / config_path.name).read_text(encoding="utf-8"))
            self.assertEqual(saved_cfg["model"]["type"], "mfnet_unetformer")
            self.assertEqual(saved_cfg["model"]["sam_checkpoint"], "/tmp/sam_vit_b.pth")
            self.assertEqual(saved_cfg["dataset"]["name"], "vaihingen")
            self.assertEqual(config_path.read_text(encoding="utf-8"), config_text)
            self.assertFalse((work_dir / "train_config.jsonc").exists())

    def test_train_entry_model_type_overrides_config_for_new_experiment(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            config_path = config_dir / "external_config.jsonc"
            config_text = '{"model": {"type": "mfnet_unetformer"}}\n'
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
                    "load_from": None,
                    "model_type": "mfnet_unetformer_dga10",
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=self._make_train_entry_config(root_dir=str(work_dir)),
                default_work_dir=work_dir,
            )

            self.assertEqual(result["default_work_dir_calls"][0]["model_name"], "mfnet_unetformer_dga10")
            self.assertEqual(result["default_work_dir_calls"][0]["seed"], 80)
            self.assertEqual(result["model_cfg"][0]["type"], "mfnet_unetformer_dga10")
            self.assertEqual(result["trainer_classes"], ["MFNetDGATrainer"])
            saved_cfg = json.loads((work_dir / config_path.name).read_text(encoding="utf-8"))
            self.assertEqual(saved_cfg["model"]["type"], "mfnet_unetformer_dga10")
            self.assertEqual(config_path.read_text(encoding="utf-8"), config_text)

    def test_train_entry_seed_overrides_config_for_new_experiment(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            config_path = config_dir / "external_config.jsonc"
            config_text = '{"seed": 80}\n'
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
                    "load_from": None,
                    "seed": 123,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=self._make_train_entry_config(root_dir=str(work_dir)),
                default_work_dir=work_dir,
            )

            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertEqual(result["default_work_dir_calls"][0]["seed"], 123)
            self.assertEqual(trainer_kwargs["cfg"]["seed"], 123)
            saved_cfg = json.loads((work_dir / config_path.name).read_text(encoding="utf-8"))
            self.assertEqual(saved_cfg["seed"], 123)
            self.assertEqual(config_path.read_text(encoding="utf-8"), config_text)

    def test_train_entry_saves_merged_config_using_child_config_name(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            base_config_path = config_dir / "base_config.jsonc"
            child_config_path = config_dir / "child_config.jsonc"
            base_cfg = self._make_train_entry_config(root_dir=str(Path(tmpdir) / "dataset_root"))
            base_cfg["train"]["batch_size"] = 2  # type: ignore[index]
            base_cfg["dataset"]["train_ids"] = ["1", "2"]  # type: ignore[index]
            base_config_path.write_text(json.dumps(base_cfg, indent=4) + "\n", encoding="utf-8")
            child_config_text = """
            {
              "extends": "base_config.jsonc",
              "seed": 80,
              "dataset": {
                "train_ids": ["3"]
              },
              "train": {
                "batch_size": 4
              }
            }
            """
            child_config_path.write_text(child_config_text, encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            args = type(
                "Args",
                (),
                {
                    "config": str(child_config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                    "model_type": "mfnet_unetformer_dga10",
                    "seed": 123,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=None,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["load_config_paths"], [str(child_config_path)])
            self.assertFalse((work_dir / base_config_path.name).exists())
            saved_config_path = work_dir / child_config_path.name
            self.assertTrue(saved_config_path.is_file())
            saved_cfg = json.loads(saved_config_path.read_text(encoding="utf-8"))
            self.assertNotIn("extends", saved_cfg)
            self.assertEqual(saved_cfg["seed"], 123)
            self.assertEqual(saved_cfg["model"]["type"], "mfnet_unetformer_dga10")
            self.assertEqual(saved_cfg["model"]["num_classes"], 6)
            self.assertEqual(saved_cfg["dataset"]["train_ids"], ["3"])
            self.assertEqual(saved_cfg["dataset"]["val_ids"], ["5"])
            self.assertEqual(saved_cfg["train"]["batch_size"], 4)
            self.assertEqual(saved_cfg["train"]["max_epochs"], 1)
            self.assertEqual(child_config_path.read_text(encoding="utf-8"), child_config_text)

    def test_train_entry_uses_dga_trainer_and_logger_for_dga_model(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "mfnet_unetformer_dga10"  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetDGATrainer"])
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], TestNetRecorderLogger)

    def test_train_entry_uses_dga_trainer_and_logger_for_dga2_model(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "mfnet_unetformer_dga20"  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetDGATrainer"])
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], TestNetRecorderLogger)

    def test_train_entry_uses_dga_trainer_and_logger_for_dga20_dgsf10_model(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "mfnet_unetformer_dga20_dgsf10"  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetDGATrainer"])
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], TestNetRecorderLogger)

    def test_train_entry_uses_base_trainer_and_recorder_logger_for_dgsf10_model(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "mfnet_unetformer_dgsf10"  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetTrainer"])
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], TestNetRecorderLogger)

    def test_train_entry_uses_dga_trainer_and_logger_for_dga3_model(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "mfnet_unetformer_dga30"  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetDGATrainer"])
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], TestNetRecorderLogger)

    def test_train_entry_uses_auxalign_trainer_for_auxalign_model(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "mfnet_unetformer_prealign_auxalign"  # type: ignore[index]
            cfg["train"]["lambda_align"] = 0.5  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetAuxAlignTrainer"])
            self.assertEqual(result["default_work_dir_calls"][0]["lambda_align"], 0.5)
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], TestNetLogger)

    def test_train_entry_uses_auxalign_trainer_and_recorder_logger_for_prealign_auxalign_dgsf10(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "mfnet_unetformer_prealign_auxalign_dgsf10"  # type: ignore[index]
            cfg["train"]["lambda_align"] = 0.5  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetAuxAlignTrainer"])
            self.assertEqual(result["default_work_dir_calls"][0]["lambda_align"], 0.5)
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], TestNetRecorderLogger)
            self.assertEqual(trainer_kwargs["cfg"]["lambda_align"], 0.5)

    def test_train_entry_uses_auxalign_dga_trainer_for_combined_model(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "mfnet_unetformer_prealign_auxalign_dga10"  # type: ignore[index]
            cfg["train"]["lambda_align"] = 0.5  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetAuxAlignDGATrainer"])
            self.assertEqual(result["default_work_dir_calls"][0]["lambda_align"], 0.5)
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], TestNetRecorderLogger)
            self.assertEqual(trainer_kwargs["cfg"]["lambda_align"], 0.5)

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
            existing_log_path = resume_dir / "train.log"
            existing_log_text = "Experiment: existing-run\n"
            existing_log_path.write_text(existing_log_text, encoding="utf-8")
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
            self.assertEqual(existing_log_path.read_text(encoding="utf-8"), existing_log_text)
            self.assertFalse((resume_dir / external_config_path.name).exists())

    def test_train_entry_rejects_model_type_override_with_resume_dir(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            resume_dir = Path(tmpdir) / "resume_work"
            resume_dir.mkdir()
            resume_config_path = resume_dir / "resume_config.json"
            resume_config_text = '{"train": {"max_epochs": 9}}\n'
            resume_config_path.write_text(resume_config_text, encoding="utf-8")
            external_config_path = Path(tmpdir) / "external_config.jsonc"
            external_config_text = '{"train": {"max_epochs": 1}}\n'
            external_config_path.write_text(external_config_text, encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "config": str(external_config_path),
                    "device": "cpu",
                    "resume_dir": str(resume_dir),
                    "resume_ckpt": None,
                    "load_from": None,
                    "model_type": "mfnet_unetformer_dga10",
                },
            )()

            with self.assertRaisesRegex(ValueError, "--model-type cannot be used with --resume-dir"):
                self._run_train_entry(
                    module=module,
                    args=args,
                    cfg=self._make_train_entry_config(root_dir=str(resume_dir)),
                    default_work_dir=None,
                )

            self.assertEqual(resume_config_path.read_text(encoding="utf-8"), resume_config_text)
            self.assertEqual(external_config_path.read_text(encoding="utf-8"), external_config_text)
            self.assertFalse((resume_dir / external_config_path.name).exists())

    def test_train_entry_rejects_seed_override_with_resume_dir(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            resume_dir = Path(tmpdir) / "resume_work"
            resume_dir.mkdir()
            resume_config_path = resume_dir / "resume_config.json"
            resume_config_text = '{"train": {"max_epochs": 9}}\n'
            resume_config_path.write_text(resume_config_text, encoding="utf-8")
            external_config_path = Path(tmpdir) / "external_config.jsonc"
            external_config_text = '{"train": {"max_epochs": 1}}\n'
            external_config_path.write_text(external_config_text, encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "config": str(external_config_path),
                    "device": "cpu",
                    "resume_dir": str(resume_dir),
                    "resume_ckpt": None,
                    "load_from": None,
                    "seed": 123,
                },
            )()

            with self.assertRaisesRegex(ValueError, "--seed cannot be used with --resume-dir"):
                self._run_train_entry(
                    module=module,
                    args=args,
                    cfg=self._make_train_entry_config(root_dir=str(resume_dir)),
                    default_work_dir=None,
                )

            self.assertEqual(resume_config_path.read_text(encoding="utf-8"), resume_config_text)
            self.assertEqual(external_config_path.read_text(encoding="utf-8"), external_config_text)
            self.assertFalse((resume_dir / external_config_path.name).exists())


if __name__ == "__main__":
    unittest.main()
