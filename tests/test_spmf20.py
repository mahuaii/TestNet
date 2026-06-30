from __future__ import annotations

import math
import unittest

import torch

from models.mfnet.modules.dsm_structure_branch10 import DSMStructureBranch10 as SplitDSMStructureBranch10
from models.mfnet.modules.spmf10 import DSMStructureBranch10 as SPMF10DSMStructureBranch10
from models.mfnet.modules.spmf20 import (
    DSMStructureBranch10 as SPMF20DSMStructureBranch10,
    MultiScaleSPMFFusion20,
    SPMFFusionBlock20,
)
from models.mfnet.modules.spmf20_fusion import (
    MultiScaleSPMFFusion20 as SplitMultiScaleSPMFFusion20,
)
from models.mfnet.modules.spmf20_fusion import (
    SPMFFusionBlock20 as SplitSPMFFusionBlock20,
)


class SPMF20FacadeTest(unittest.TestCase):
    def test_facades_reuse_split_structure_and_fusion_implementations(self) -> None:
        self.assertIs(SPMF10DSMStructureBranch10, SplitDSMStructureBranch10)
        self.assertIs(SPMF20DSMStructureBranch10, SplitDSMStructureBranch10)
        self.assertIs(SPMFFusionBlock20, SplitSPMFFusionBlock20)
        self.assertIs(MultiScaleSPMFFusion20, SplitMultiScaleSPMFFusion20)


class SPMFFusionBlock20Test(unittest.TestCase):
    def test_forward_returns_rgb_shaped_output(self) -> None:
        module = SPMFFusionBlock20(channels=8, structure_channels=5, hidden_dim=12)
        rgb = torch.randn(2, 8, 7, 9)
        dsm = torch.randn(2, 8, 7, 9)
        structure = torch.randn(2, 5, 7, 9)

        output = module(rgb, dsm, structure)

        self.assertEqual(output.shape, rgb.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_uses_structure_conditioned_affine_and_independent_evidence_heads(self) -> None:
        module = SPMFFusionBlock20(channels=8, structure_channels=5, hidden_dim=12)

        self.assertEqual(module.rgb_affine.in_channels, 12)
        self.assertEqual(module.rgb_affine.out_channels, 24)
        self.assertEqual(module.dsm_affine.in_channels, 12)
        self.assertEqual(module.dsm_affine.out_channels, 24)
        self.assertEqual(module.rgb_evidence_head[0][0].in_channels, 24)
        self.assertEqual(module.rgb_evidence_head[-1].out_channels, 8)
        self.assertEqual(module.dsm_evidence_head[0][0].in_channels, 24)
        self.assertEqual(module.dsm_evidence_head[-1].out_channels, 8)
        self.assertIsNot(module.rgb_affine, module.dsm_affine)
        self.assertIsNot(module.rgb_evidence_head, module.dsm_evidence_head)

        for forbidden_name in (
            "fusion_gate_generator",
            "prior_generator",
            "dsm_compensation",
            "dsm_message",
            "rgb_message",
            "residual_scale",
            "gamma",
        ):
            self.assertFalse(hasattr(module, forbidden_name))

    def test_zero_initialization_produces_exact_modality_average(self) -> None:
        module = SPMFFusionBlock20(channels=8, structure_channels=5, hidden_dim=12)
        rgb = torch.randn(2, 8, 7, 9)
        dsm = torch.randn(2, 8, 7, 9)
        structure = torch.randn(2, 5, 7, 9)

        output = module(rgb, dsm, structure)

        for affine in (module.rgb_affine, module.dsm_affine):
            self.assertEqual(torch.count_nonzero(affine.weight).item(), 0)
            self.assertEqual(torch.count_nonzero(affine.bias).item(), 0)
        for evidence_head in (module.rgb_evidence_head, module.dsm_evidence_head):
            self.assertEqual(torch.count_nonzero(evidence_head[-1].weight).item(), 0)
            self.assertEqual(torch.count_nonzero(evidence_head[-1].bias).item(), 0)
        self.assertTrue(torch.equal(output, 0.5 * (rgb + dsm)))

    def test_softmax_evidence_competes_between_modalities(self) -> None:
        module = SPMFFusionBlock20(channels=3, structure_channels=2, hidden_dim=4)
        with torch.no_grad():
            module.rgb_evidence_head[-1].bias.fill_(math.log(3.0))
            module.dsm_evidence_head[-1].bias.zero_()
        rgb = torch.randn(2, 3, 5, 7)
        dsm = torch.randn(2, 3, 5, 7)
        structure = torch.randn(2, 2, 5, 7)

        output = module(rgb, dsm, structure)

        self.assertTrue(torch.allclose(output, 0.75 * rgb + 0.25 * dsm, atol=1e-6, rtol=1e-6))

    def test_rejects_invalid_inputs(self) -> None:
        module = SPMFFusionBlock20(channels=8, structure_channels=5, hidden_dim=12)
        rgb = torch.randn(2, 8, 7, 9)
        dsm = torch.randn(2, 8, 7, 9)
        structure = torch.randn(2, 5, 7, 9)

        with self.assertRaises(ValueError):
            module(rgb[:, 0], dsm, structure)
        with self.assertRaises(ValueError):
            module(torch.randn(2, 4, 7, 9), dsm, structure)
        with self.assertRaises(ValueError):
            module(rgb, torch.randn(2, 8, 6, 9), structure)
        with self.assertRaises(ValueError):
            module(rgb, dsm, torch.randn(2, 4, 7, 9))
        with self.assertRaises(ValueError):
            module(rgb, dsm, torch.randn(2, 5, 6, 9))

    def test_backward_produces_finite_input_and_parameter_gradients(self) -> None:
        module = SPMFFusionBlock20(channels=8, structure_channels=5, hidden_dim=12)
        rgb = torch.randn(2, 8, 7, 9, requires_grad=True)
        dsm = torch.randn(2, 8, 7, 9, requires_grad=True)
        structure = torch.randn(2, 5, 7, 9, requires_grad=True)

        output = module(rgb, dsm, structure)
        output.square().mean().backward()

        for feature in (rgb, dsm, structure):
            self.assertIsNotNone(feature.grad)
            self.assertTrue(torch.isfinite(feature.grad).all())
        for parameter in module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


class MultiScaleSPMFFusion20Test(unittest.TestCase):
    def test_forward_returns_four_rgb_shaped_outputs(self) -> None:
        module = MultiScaleSPMFFusion20(
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
        self.assertEqual(len(module.blocks), 4)
        self.assertEqual(len({id(block) for block in module.blocks}), 4)
        for output, rgb, dsm in zip(outputs, rgb_feats, dsm_feats):
            self.assertEqual(output.shape, rgb.shape)
            self.assertTrue(torch.equal(output, 0.5 * (rgb + dsm)))

    def test_rejects_invalid_feature_sequence_and_constructor_lengths(self) -> None:
        module = MultiScaleSPMFFusion20(channels=8, hidden_dim=8)
        features = tuple(torch.randn(2, 8, 4, 4) for _ in range(4))

        with self.assertRaises(ValueError):
            module(features[:3], features, features)
        with self.assertRaises(ValueError):
            module(features, features[:3], features)
        with self.assertRaises(ValueError):
            module(features, features, features[:3])
        with self.assertRaises(TypeError):
            module(object(), features, features)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MultiScaleSPMFFusion20(channels=(8, 8, 8), hidden_dim=8)
        with self.assertRaises(ValueError):
            MultiScaleSPMFFusion20(
                channels=8,
                structure_channels=(8, 8, 8),
                hidden_dim=8,
            )


if __name__ == "__main__":
    unittest.main()
