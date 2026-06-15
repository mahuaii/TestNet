from __future__ import annotations

import importlib
import sys
import types
import unittest

import torch

from models.mfnet.modules import (
    DSMStructureBranch11,
    MultiScaleStructurePriorModulatedFusion11,
    StructurePriorModulatedFusionBlock11,
)
from models.mfnet.modules.dsm_structure_branch11 import DSMStructureBranch11 as SplitDSMStructureBranch11
from models.mfnet.modules.spmf11 import (
    DSMStructureBranch11 as FacadeDSMStructureBranch11,
)
from models.mfnet.modules.spmf11 import (
    MultiScaleSPMF11,
    SPMFBlock11,
)
from models.mfnet.modules.spmf11_fusion import (
    MultiScaleStructurePriorModulatedFusion11 as SplitMultiScaleSPMF11,
)
from models.mfnet.modules.spmf11_fusion import (
    StructurePriorModulatedFusionBlock11 as SplitSPMFBlock11,
)


class SPMF11FacadeTest(unittest.TestCase):
    def test_facade_reuses_split_implementations(self) -> None:
        self.assertIs(FacadeDSMStructureBranch11, SplitDSMStructureBranch11)
        self.assertIs(StructurePriorModulatedFusionBlock11, SplitSPMFBlock11)
        self.assertIs(MultiScaleStructurePriorModulatedFusion11, SplitMultiScaleSPMF11)
        self.assertIs(SPMFBlock11, StructurePriorModulatedFusionBlock11)
        self.assertIs(MultiScaleSPMF11, MultiScaleStructurePriorModulatedFusion11)


class SPMF11BuildTest(unittest.TestCase):
    def test_build_model_dispatches_to_spmf11(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerSPMF11:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerSPMF11 = FakeUNetFormerSPMF11
        original_mfnet_module = sys.modules.get("models.mfnet")
        try:
            sys.modules["models.mfnet"] = fake_mfnet_module
            model = build_module.build_model(
                {
                    "type": "testnet_spmf11",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["spmf11"],
                }
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

        self.assertIsInstance(model, FakeUNetFormerSPMF11)
        self.assertEqual(
            captured_kwargs,
            [
                {
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["spmf11"],
                }
            ],
        )
        self.assertIn("testnet_spmf11", build_module.AVAILABLE_MODEL_TYPES)


class UNetFormerSPMF11Test(unittest.TestCase):
    def test_forward_routes_sam_taps_through_structure_and_spmf11(self) -> None:
        from models.mfnet.UNetFormer_MMSAM_spmf11 import UNetFormerSPMF11

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

        class SPMF11(torch.nn.Module):
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

        model = UNetFormerSPMF11.__new__(UNetFormerSPMF11)
        torch.nn.Module.__init__(model)
        model.image_encoder = ImageEncoder()
        model.spmf11_indexes = [0, 1, 2, 3]
        model.fpn1x = model.fpn2x = model.fpn3x = model.fpn4x = torch.nn.Identity()
        model.fpn1y = model.fpn2y = model.fpn3y = model.fpn4y = torch.nn.Identity()
        model.structure_branch11 = StructureBranch()
        model.spmf11 = SPMF11()
        model.decoder = Decoder()
        raw_dsm = torch.rand(2, 8, 8)

        output = model(torch.rand(2, 3, 8, 8), raw_dsm)

        self.assertEqual(output.shape, (2, 6, 8, 8))
        self.assertTrue(torch.equal(model.structure_branch11.dsm, raw_dsm.unsqueeze(1)))
        self.assertEqual(len(model.structure_branch11.taps or ()), 4)
        self.assertIsNotNone(model.spmf11.inputs)
        self.assertIsNotNone(model.decoder.inputs)


class DSMStructureBranch11Test(unittest.TestCase):
    def _make_taps(self, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            torch.randn(2, channel, 4, 5, requires_grad=requires_grad)
            for channel in (3, 4, 5, 6)
        )  # type: ignore[return-value]

    def test_returns_four_structure_features(self) -> None:
        module = DSMStructureBranch11(
            tap_channels=(3, 4, 5, 6),
            structure_channels=(8, 8, 8, 8),
            output_channels=12,
            similarity_kernel_size=3,
        )

        outputs = module(torch.rand(2, 64, 80), self._make_taps())

        self.assertEqual(
            [output.shape for output in outputs],
            [
                (2, 12, 16, 20),
                (2, 12, 8, 10),
                (2, 12, 4, 5),
                (2, 12, 2, 2),
            ],
        )

    def test_structure_input_preserves_pre_normalized_dsm_values(self) -> None:
        module = DSMStructureBranch11(
            tap_channels=(3, 4, 5, 6),
            structure_channels=(8, 8, 8, 8),
            output_channels=12,
            similarity_kernel_size=3,
        )
        dsm = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]])

        structure_input = module._make_structure_input(dsm)

        self.assertTrue(torch.equal(structure_input[:, :1], dsm))

    def test_backward_keeps_taps_detached_and_produces_finite_gradients(self) -> None:
        module = DSMStructureBranch11(
            tap_channels=(3, 4, 5, 6),
            structure_channels=(8, 8, 8, 8),
            output_channels=12,
            similarity_kernel_size=3,
        )
        dsm = torch.rand(2, 1, 64, 80, requires_grad=True)
        taps = self._make_taps(requires_grad=True)

        sum(output.square().mean() for output in module(dsm, taps)).backward()

        self.assertIsNotNone(dsm.grad)
        self.assertTrue(torch.isfinite(dsm.grad).all())
        for tap in taps:
            self.assertIsNone(tap.grad)
        for parameter in module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


class StructurePriorModulatedFusionBlock11Test(unittest.TestCase):
    def test_uses_independent_nonzero_initialized_modality_gates(self) -> None:
        module = StructurePriorModulatedFusionBlock11(
            channels=8,
            structure_channels=5,
            hidden_dim=12,
            gate_init_std=1e-3,
        )
        rgb_output = module.rgb_gate_generator[-1]
        dsm_output = module.dsm_gate_generator[-1]

        self.assertIsNot(module.rgb_gate_generator, module.dsm_gate_generator)
        self.assertGreater(torch.count_nonzero(rgb_output.weight).item(), 0)
        self.assertGreater(torch.count_nonzero(dsm_output.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(rgb_output.bias).item(), 0)
        self.assertEqual(torch.count_nonzero(dsm_output.bias).item(), 0)
        self.assertLess(rgb_output.weight.std().item(), 0.01)
        self.assertLess(dsm_output.weight.std().item(), 0.01)

    def test_independent_gate_biases_do_not_form_a_complementary_pair(self) -> None:
        module = StructurePriorModulatedFusionBlock11(channels=3, structure_channels=2, hidden_dim=4)
        with torch.no_grad():
            module.rgb_gate_generator[-1].weight.zero_()
            module.rgb_gate_generator[-1].bias.fill_(torch.logit(torch.tensor(0.75)))
            module.dsm_gate_generator[-1].weight.zero_()
            module.dsm_gate_generator[-1].bias.fill_(torch.logit(torch.tensor(0.75)))
        rgb = torch.randn(2, 3, 5, 7)
        dsm = torch.randn(2, 3, 5, 7)
        structure = torch.randn(2, 2, 5, 7)

        output = module(rgb, dsm, structure)

        self.assertTrue(torch.allclose(output, 0.75 * rgb + 0.75 * dsm, atol=1e-6, rtol=1e-6))

    def test_first_backward_reaches_gate_projections(self) -> None:
        module = StructurePriorModulatedFusionBlock11(channels=8, structure_channels=5, hidden_dim=12)
        rgb = torch.randn(2, 8, 7, 9, requires_grad=True)
        dsm = torch.randn(2, 8, 7, 9, requires_grad=True)
        structure = torch.randn(2, 5, 7, 9, requires_grad=True)

        module(rgb, dsm, structure).square().mean().backward()

        for projection in (module.rgb_projection, module.dsm_projection, module.structure_projection):
            grad = projection[0].weight.grad
            self.assertIsNotNone(grad)
            self.assertGreater(torch.count_nonzero(grad).item(), 0)
            self.assertTrue(torch.isfinite(grad).all())

    def test_rejects_nonpositive_gate_init_std(self) -> None:
        with self.assertRaises(ValueError):
            StructurePriorModulatedFusionBlock11(gate_init_std=0.0)


class MultiScaleStructurePriorModulatedFusion11Test(unittest.TestCase):
    def test_forward_returns_four_rgb_shaped_outputs(self) -> None:
        module = MultiScaleStructurePriorModulatedFusion11(
            channels=(8, 10, 12, 14),
            structure_channels=(3, 4, 5, 6),
            hidden_dim=(8, 8, 8, 8),
        )
        sizes = ((16, 20), (8, 10), (4, 5), (2, 3))
        rgb_feats = tuple(torch.randn(2, channel, *size) for channel, size in zip((8, 10, 12, 14), sizes))
        dsm_feats = tuple(torch.randn(2, channel, *size) for channel, size in zip((8, 10, 12, 14), sizes))
        structure_feats = tuple(torch.randn(2, channel, *size) for channel, size in zip((3, 4, 5, 6), sizes))

        outputs = module(rgb_feats, dsm_feats, structure_feats)

        self.assertEqual(len(outputs), 4)
        for output, rgb in zip(outputs, rgb_feats):
            self.assertEqual(output.shape, rgb.shape)
            self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
