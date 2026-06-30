from __future__ import annotations

import importlib
import sys
import types
import unittest

import torch

from models.mfnet.modules import (
    DSMStructureBranch13,
    MultiScaleSPMFFusion22,
    SPMFFusionBlock22,
)
from models.mfnet.modules.dsm_structure_branch13 import DSMStructureBranch13 as SplitDSMStructureBranch13
from models.mfnet.modules.spmf20_fusion import (
    MultiScaleSPMFFusion20,
    SPMFFusionBlock20,
)
from models.mfnet.modules.spmf22 import DSMStructureBranch13 as FacadeDSMStructureBranch13
from utils import IntermediateStatsRecorder


class SPMF22FacadeTest(unittest.TestCase):
    def test_facade_uses_branch13_and_unchanged_spmf20_fusion(self) -> None:
        self.assertIs(FacadeDSMStructureBranch13, SplitDSMStructureBranch13)
        self.assertIs(SPMFFusionBlock22, SPMFFusionBlock20)
        self.assertIs(MultiScaleSPMFFusion22, MultiScaleSPMFFusion20)


class DSMStructureBranch13Test(unittest.TestCase):
    def _make_taps(self, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            torch.randn(2, channel, 4, 5, requires_grad=requires_grad)
            for channel in (3, 4, 5, 6)
        )  # type: ignore[return-value]

    def test_scale_from_logits_is_bounded_and_centered_at_one(self) -> None:
        logits = torch.tensor([[[[-100.0, 0.0, 100.0]]]])

        scale = DSMStructureBranch13._scale_from_logits(logits)

        self.assertTrue(torch.all(scale >= 0.5))
        self.assertTrue(torch.all(scale <= 1.5))
        self.assertAlmostEqual(scale[0, 0, 0, 1].item(), 1.0, places=6)

    def test_zero_logits_modulation_returns_projected_geometry(self) -> None:
        projected = torch.randn(2, 12, 7, 9)
        scale = DSMStructureBranch13._scale_from_logits(torch.zeros_like(projected))

        output = DSMStructureBranch13._modulate_structure(projected, scale)

        self.assertTrue(torch.equal(output, projected))

    def test_scale_heads_zero_initialize_last_layer_and_preserve_spatial_shape(self) -> None:
        module = DSMStructureBranch13(
            tap_channels=(3, 4, 5, 6),
            structure_channels=(8, 8, 8, 8),
            output_channels=12,
            similarity_kernel_size=3,
        )

        self.assertFalse(hasattr(module, "confidence_alphas"))
        self.assertFalse(hasattr(module, "confidence_generators"))
        for scale_head in module.scale_heads:
            self.assertTrue(torch.equal(scale_head.out.weight, torch.zeros_like(scale_head.out.weight)))
            self.assertTrue(torch.equal(scale_head.out.bias, torch.zeros_like(scale_head.out.bias)))

        z = module.scale_heads[0](torch.randn(2, 8, 7, 9))
        self.assertEqual(z.shape, (2, 12, 7, 9))
        self.assertTrue(torch.equal(z, torch.zeros_like(z)))

    def test_returns_four_structure_features_and_keeps_taps_detached(self) -> None:
        module = DSMStructureBranch13(
            tap_channels=(3, 4, 5, 6),
            structure_channels=(8, 8, 8, 8),
            output_channels=12,
            similarity_kernel_size=3,
        )
        dsm = torch.randn(2, 1, 64, 80, requires_grad=True)
        taps = self._make_taps(requires_grad=True)

        outputs = module(dsm, taps)
        sum(output.square().mean() for output in outputs).backward()

        self.assertEqual(
            [output.shape for output in outputs],
            [
                (2, 12, 16, 20),
                (2, 12, 8, 10),
                (2, 12, 4, 5),
                (2, 12, 2, 2),
            ],
        )
        self.assertIsNotNone(dsm.grad)
        self.assertTrue(torch.isfinite(dsm.grad).all())
        for tap in taps:
            self.assertIsNone(tap.grad)

    def test_records_scale_structure_and_structure_norm_statistics(self) -> None:
        module = DSMStructureBranch13(
            tap_channels=(3, 4, 5, 6),
            structure_channels=(8, 8, 8, 8),
            output_channels=12,
            similarity_kernel_size=3,
        )
        module.intermediate_stats = IntermediateStatsRecorder()
        module.intermediate_stats_prefix = "spmf22/structure"

        module(torch.randn(2, 1, 64, 80), self._make_taps())

        snapshot = module.intermediate_stats.snapshot()
        for base in (
            "spmf22/structure/scale/scale1",
            "spmf22/structure/structure/structure1",
        ):
            for suffix in ("mean", "std", "var", "min", "max"):
                self.assertIn(f"{base}_{suffix}", snapshot)
        self.assertIn("spmf22/structure/structure/structure1_norm", snapshot)
        self.assertAlmostEqual(snapshot["spmf22/structure/scale/scale1_mean"], 1.0, places=6)
        self.assertNotIn("spmf22/structure/alpha/alpha1_mean", snapshot)
        self.assertNotIn("spmf22/structure/confidence/confidence1_mean", snapshot)


class SPMF22BuildTest(unittest.TestCase):
    def test_build_model_dispatches_to_spmf22(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerSPMF22:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerSPMF22 = FakeUNetFormerSPMF22
        original_mfnet_module = sys.modules.get("models.mfnet")
        try:
            sys.modules["models.mfnet"] = fake_mfnet_module
            model = build_module.build_model(
                {
                    "type": "testnet_spmf22",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["structure13"],
                }
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

        self.assertIsInstance(model, FakeUNetFormerSPMF22)
        self.assertEqual(
            captured_kwargs,
            [
                {
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["structure13"],
                }
            ],
        )
        self.assertIn("testnet_spmf22", build_module.AVAILABLE_MODEL_TYPES)


class UNetFormerSPMF22Test(unittest.TestCase):
    def test_forward_routes_sam_taps_through_structure13_and_spmf22_only(self) -> None:
        from models.mfnet.UNetFormer_MMSAM_spmf22 import UNetFormerSPMF22

        class PatchEmbed(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                return x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1).permute(0, 2, 3, 1)

        class Block(torch.nn.Module):
            window_size = 0

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return x + 1.0, y + 2.0

        class ImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch_embed = PatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(Block() for _ in range(4))
                self.neck = torch.nn.Identity()

        class StructureBranch(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dsm: torch.Tensor | None = None
                self.taps: tuple[torch.Tensor, ...] | None = None

            def forward(
                self,
                dsm: torch.Tensor,
                taps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                self.dsm = dsm
                self.taps = taps
                return tuple(torch.full_like(tap, 3.0) for tap in taps)  # type: ignore[return-value]

        class SPMF22(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: tuple[tuple[torch.Tensor, ...], ...] | None = None

            def forward(
                self,
                rgb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                dsm: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                structure: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                self.inputs = (rgb, dsm, structure)
                return tuple(rgb_feature + dsm_feature for rgb_feature, dsm_feature in zip(rgb, dsm))

        class Decoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: tuple[torch.Tensor, ...] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                self.inputs = (res1, res2, res3, res4)
                return torch.zeros(res1.shape[0], 6, h, w)

        model = UNetFormerSPMF22.__new__(UNetFormerSPMF22)
        torch.nn.Module.__init__(model)
        model.image_encoder = ImageEncoder()
        model.spmf22_indexes = [0, 1, 2, 3]
        model.fpn1x = model.fpn2x = model.fpn3x = model.fpn4x = torch.nn.Identity()
        model.fpn1y = model.fpn2y = model.fpn3y = model.fpn4y = torch.nn.Identity()
        model.structure_branch13 = StructureBranch()
        model.spmf_fusion22 = SPMF22()
        model.decoder = Decoder()
        raw_dsm = torch.rand(2, 8, 8)

        output = model(torch.rand(2, 3, 8, 8), raw_dsm)

        self.assertEqual(output.shape, (2, 6, 8, 8))
        self.assertTrue(torch.equal(model.structure_branch13.dsm, raw_dsm.unsqueeze(1)))
        self.assertEqual(len(model.structure_branch13.taps or ()), 4)
        self.assertIsNotNone(model.spmf_fusion22.inputs)
        self.assertIsNotNone(model.decoder.inputs)
        self.assertFalse(hasattr(model, "structure_branch12"))
        self.assertFalse(hasattr(model, "spmf_fusion21"))


if __name__ == "__main__":
    unittest.main()
