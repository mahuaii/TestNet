from __future__ import annotations

import unittest

import torch

from models.mfnet.modules import (
    DSMStructureBranch10,
    MultiScaleStructurePriorModulatedFusion10,
    StructurePriorModulatedFusionBlock10,
)


class DSMStructureBranch10Test(unittest.TestCase):
    def _make_taps(self, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            torch.randn(2, channel, 4, 5, requires_grad=requires_grad)
            for channel in (3, 4, 5, 6)
        )  # type: ignore[return-value]

    def test_accepts_three_dimensional_dsm_and_returns_four_structure_features(self) -> None:
        module = DSMStructureBranch10(
            tap_channels=(3, 4, 5, 6),
            structure_channels=(8, 8, 8, 8),
            output_channels=12,
            similarity_kernel_size=3,
        )
        dsm = torch.randn(2, 64, 80)

        outputs = module(dsm, self._make_taps())

        self.assertEqual(len(outputs), 4)
        for output, expected_size in zip(outputs, ((16, 20), (8, 10), (4, 5), (2, 2))):
            self.assertEqual(output.shape, (2, 12, *expected_size))
            self.assertTrue(torch.isfinite(output).all())

    def test_accepts_four_dimensional_dsm(self) -> None:
        module = DSMStructureBranch10(
            tap_channels=(3, 4, 5, 6),
            structure_channels=(8, 8, 8, 8),
            output_channels=12,
            similarity_kernel_size=3,
        )
        dsm = torch.randn(2, 1, 64, 80)

        outputs = module(dsm, self._make_taps())

        self.assertEqual([output.shape for output in outputs], [
            (2, 12, 16, 20),
            (2, 12, 8, 10),
            (2, 12, 4, 5),
            (2, 12, 2, 2),
        ])

    def test_rejects_invalid_inputs(self) -> None:
        module = DSMStructureBranch10(
            tap_channels=(3, 4, 5, 6),
            structure_channels=(8, 8, 8, 8),
            output_channels=12,
            similarity_kernel_size=3,
        )
        dsm = torch.randn(2, 64, 80)
        taps = self._make_taps()

        with self.assertRaises(ValueError):
            module(torch.randn(2, 2, 64, 80), taps)
        with self.assertRaises(ValueError):
            module(dsm, taps[:3])
        with self.assertRaises(ValueError):
            module(dsm, (torch.randn(2, 2, 4, 5), *taps[1:]))
        with self.assertRaises(ValueError):
            module(dsm, (torch.randn(3, 3, 4, 5), *taps[1:]))
        with self.assertRaises(TypeError):
            module(dsm, object())  # type: ignore[arg-type]

    def test_backward_produces_finite_input_and_parameter_gradients(self) -> None:
        module = DSMStructureBranch10(
            tap_channels=(3, 4, 5, 6),
            structure_channels=(8, 8, 8, 8),
            output_channels=12,
            similarity_kernel_size=3,
        )
        dsm = torch.randn(2, 1, 64, 80, requires_grad=True)
        taps = self._make_taps(requires_grad=True)

        outputs = module(dsm, taps)
        loss = sum(output.square().mean() for output in outputs)
        loss.backward()

        self.assertIsNotNone(dsm.grad)
        self.assertTrue(torch.isfinite(dsm.grad).all())
        for tap in taps:
            self.assertIsNone(tap.grad)
        for parameter in module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_confidence_generators_match_projected_output_channels(self) -> None:
        module = DSMStructureBranch10(
            tap_channels=(3, 4, 5, 6),
            structure_channels=(8, 10, 12, 14),
            output_channels=12,
            similarity_kernel_size=3,
        )

        for confidence_generator, structure_channel in zip(module.confidence_generators, (8, 10, 12, 14)):
            self.assertEqual(confidence_generator[0][0].in_channels, structure_channel)
            self.assertEqual(confidence_generator[1].out_channels, 12)


class StructurePriorModulatedFusionBlock10Test(unittest.TestCase):
    def test_forward_returns_rgb_shaped_output(self) -> None:
        module = StructurePriorModulatedFusionBlock10(channels=8, structure_channels=5, hidden_dim=12)
        rgb = torch.randn(2, 8, 7, 9)
        dsm = torch.randn(2, 8, 7, 9)
        structure = torch.randn(2, 5, 7, 9)

        output = module(rgb, dsm, structure)

        self.assertEqual(output.shape, rgb.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_uses_three_projected_inputs_to_generate_feature_channel_gate(self) -> None:
        module = StructurePriorModulatedFusionBlock10(channels=8, structure_channels=5, hidden_dim=12)

        self.assertEqual(module.fusion_gate_generator[0][0].in_channels, 36)
        self.assertEqual(module.fusion_gate_generator[1].out_channels, 8)
        self.assertFalse(hasattr(module, "structure_gate"))
        self.assertFalse(hasattr(module, "prior_generator"))
        self.assertFalse(hasattr(module, "dsm_compensation"))
        self.assertFalse(hasattr(module, "gamma"))

    def test_initial_gate_logits_are_zero_and_output_is_modality_average(self) -> None:
        module = StructurePriorModulatedFusionBlock10(channels=8, structure_channels=5, hidden_dim=12)
        gate_output = module.fusion_gate_generator[1]
        rgb = torch.randn(2, 8, 7, 9)
        dsm = torch.randn(2, 8, 7, 9)
        structure = torch.randn(2, 5, 7, 9)

        output = module(rgb, dsm, structure)

        self.assertTrue(torch.count_nonzero(gate_output.weight) == 0)
        self.assertTrue(torch.count_nonzero(gate_output.bias) == 0)
        self.assertTrue(torch.allclose(output, 0.5 * (rgb + dsm)))

    def test_rejects_invalid_inputs(self) -> None:
        module = StructurePriorModulatedFusionBlock10(channels=8, structure_channels=5, hidden_dim=12)
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
        module = StructurePriorModulatedFusionBlock10(channels=8, structure_channels=5, hidden_dim=12)
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


class MultiScaleStructurePriorModulatedFusion10Test(unittest.TestCase):
    def test_forward_returns_four_rgb_shaped_outputs(self) -> None:
        module = MultiScaleStructurePriorModulatedFusion10(
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

    def test_rejects_invalid_feature_sequence_lengths(self) -> None:
        module = MultiScaleStructurePriorModulatedFusion10(channels=8, hidden_dim=8)
        rgb_feats = tuple(torch.randn(2, 8, 4, 4) for _ in range(4))
        dsm_feats = tuple(torch.randn(2, 8, 4, 4) for _ in range(4))
        structure_feats = tuple(torch.randn(2, 8, 4, 4) for _ in range(4))

        with self.assertRaises(ValueError):
            module(rgb_feats[:3], dsm_feats, structure_feats)
        with self.assertRaises(ValueError):
            module(rgb_feats, dsm_feats[:3], structure_feats)
        with self.assertRaises(ValueError):
            module(rgb_feats, dsm_feats, structure_feats[:3])
        with self.assertRaises(TypeError):
            module(object(), dsm_feats, structure_feats)  # type: ignore[arg-type]

    def test_rejects_invalid_constructor_lengths(self) -> None:
        with self.assertRaises(ValueError):
            MultiScaleStructurePriorModulatedFusion10(channels=(8, 8, 8), hidden_dim=8)
        with self.assertRaises(ValueError):
            MultiScaleStructurePriorModulatedFusion10(channels=8, structure_channels=(8, 8, 8), hidden_dim=8)


if __name__ == "__main__":
    unittest.main()
