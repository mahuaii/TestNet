from __future__ import annotations

import unittest

import torch

from models.mfnet.modules import DGFM01


class DGFM01Test(unittest.TestCase):
    def test_forward_accepts_testnet_bhwc_and_returns_bchw(self) -> None:
        module = DGFM01(dims=16)
        rgb = torch.randn(2, 4, 6, 16)
        dsm = torch.randn(2, 4, 6, 16)

        output = module(rgb, dsm)

        self.assertEqual(output.shape, (2, 16, 4, 6))
        self.assertTrue(torch.isfinite(output).all())

    def test_initial_gate_returns_unprojected_average(self) -> None:
        torch.manual_seed(0)
        module = DGFM01(dims=8)
        rgb = torch.randn(2, 3, 4, 8)
        dsm = torch.randn(2, 3, 4, 8)

        output = module(rgb, dsm)

        expected = 0.5 * rgb.permute(0, 3, 1, 2).contiguous()
        expected = expected + 0.5 * dsm.permute(0, 3, 1, 2).contiguous()
        self.assertTrue(torch.allclose(output, expected))
        self.assertFalse(hasattr(module, "input_norm"))
        self.assertFalse(hasattr(module, "output_norm"))
        self.assertFalse(hasattr(module, "output_proj"))
        self.assertFalse(hasattr(module, "scale_adapter"))

    def test_core_structure_matches_mobanet_gate(self) -> None:
        module = DGFM01(dims=16, ratio=0.25)

        self.assertIsInstance(module.rgb_reduce, torch.nn.Conv2d)
        self.assertEqual(module.rgb_reduce.in_channels, 16)
        self.assertEqual(module.rgb_reduce.out_channels, 32)
        self.assertEqual(module.rgb_reduce.kernel_size, (1, 1))
        self.assertFalse(module.rgb_reduce.bias is not None)

        self.assertIsInstance(module.dsm_reduce, torch.nn.Conv2d)
        self.assertEqual(module.dsm_reduce.in_channels, 16)
        self.assertEqual(module.dsm_reduce.out_channels, 32)
        self.assertEqual(module.dsm_reduce.kernel_size, (1, 1))
        self.assertFalse(module.dsm_reduce.bias is not None)

        self.assertIsInstance(module.gate_net[0], torch.nn.Conv2d)
        self.assertEqual(module.gate_net[0].in_channels, 96)
        self.assertEqual(module.gate_net[0].out_channels, 32)
        self.assertIsInstance(module.gate_net[1], torch.nn.GroupNorm)
        self.assertEqual(module.gate_net[1].num_groups, 1)
        self.assertIsInstance(module.gate_net[2], torch.nn.GELU)
        self.assertIsInstance(module.gate_net[3], torch.nn.Conv2d)
        self.assertEqual(module.gate_net[3].in_channels, 32)
        self.assertEqual(module.gate_net[3].out_channels, 16)
        self.assertIsInstance(module.gate_net[4], torch.nn.Sigmoid)
        self.assertTrue(torch.equal(module.gate_net[3].weight, torch.zeros_like(module.gate_net[3].weight)))
        self.assertTrue(torch.equal(module.gate_net[3].bias, torch.zeros_like(module.gate_net[3].bias)))

    def test_rejects_mismatched_shapes(self) -> None:
        module = DGFM01(dims=8)

        with self.assertRaises(ValueError):
            module(torch.randn(2, 3, 4, 8), torch.randn(2, 3, 5, 8))

    def test_rejects_wrong_channel_count(self) -> None:
        module = DGFM01(dims=8)

        with self.assertRaises(ValueError):
            module(torch.randn(2, 3, 4, 7), torch.randn(2, 3, 4, 7))

    def test_backward_produces_finite_input_and_parameter_gradients(self) -> None:
        module = DGFM01(dims=8)
        rgb = torch.randn(2, 3, 4, 8, requires_grad=True)
        dsm = torch.randn(2, 3, 4, 8, requires_grad=True)

        output = module(rgb, dsm)
        loss = output.square().mean()
        loss.backward()

        self.assertIsNotNone(rgb.grad)
        self.assertIsNotNone(dsm.grad)
        self.assertTrue(torch.isfinite(rgb.grad).all())
        self.assertTrue(torch.isfinite(dsm.grad).all())

        trainable_parameters = [param for param in module.parameters() if param.requires_grad]
        self.assertGreater(len(trainable_parameters), 0)
        for param in trainable_parameters:
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())


if __name__ == "__main__":
    unittest.main()
