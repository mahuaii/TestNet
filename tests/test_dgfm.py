from __future__ import annotations

import unittest

import torch

from models.mfnet.modules import DGFM, DGFMScaleAdapter


class DGFMTest(unittest.TestCase):
    def test_forward_accepts_bhwc_and_returns_default_decoder_ready_shapes(self) -> None:
        module = DGFM(dims=16)
        rgb = torch.randn(2, 4, 6, 16)
        dsm = torch.randn(2, 4, 6, 16)

        output = module(rgb, dsm)

        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 4)
        for actual, expected_size in zip(output, [(16, 24), (8, 12), (4, 6), (2, 3)]):
            self.assertEqual(actual.shape, (2, 256, *expected_size))
            self.assertTrue(torch.isfinite(actual).all())

    def test_forward_uses_configurable_output_channels(self) -> None:
        module = DGFM(dims=16, out_channels=12)
        rgb = torch.randn(2, 4, 6, 16)
        dsm = torch.randn(2, 4, 6, 16)

        output = module(rgb, dsm)

        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 4)
        for actual, expected_size in zip(output, [(16, 24), (8, 12), (4, 6), (2, 3)]):
            self.assertEqual(actual.shape, (2, 12, *expected_size))
            self.assertTrue(torch.isfinite(actual).all())

    def test_scale_adapter_rejects_invalid_channel_count(self) -> None:
        adapter = DGFMScaleAdapter(channels=12)

        with self.assertRaises(ValueError):
            adapter(torch.randn(2, 8, 4, 6))

    def test_scale_adapter_uses_resize_conv_branches(self) -> None:
        adapter = DGFMScaleAdapter(channels=12)

        self.assertFalse(any(isinstance(module, torch.nn.ConvTranspose2d) for module in adapter.modules()))
        self.assertFalse(any(isinstance(module, torch.nn.MaxPool2d) for module in adapter.modules()))
        self.assertTrue(any(isinstance(module, torch.nn.Conv2d) and module.groups == 12 for module in adapter.modules()))

    def test_initial_gate_returns_projected_post_norm_average(self) -> None:
        torch.manual_seed(0)
        module = DGFM(dims=8, out_channels=12)
        rgb = torch.randn(2, 3, 4, 8)
        dsm = torch.randn(2, 3, 4, 8)

        output = module(rgb, dsm)

        with torch.no_grad():
            rgb_norm = module.input_norm(rgb).permute(0, 3, 1, 2).contiguous()
            dsm_norm = module.input_norm(dsm).permute(0, 3, 1, 2).contiguous()
            fused = 0.5 * rgb_norm + 0.5 * dsm_norm
            fused = module.output_norm(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            expected = module.scale_adapter(module.output_proj(fused))

        for actual, expected_scale in zip(output, expected):
            self.assertTrue(torch.allclose(actual, expected_scale))

    def test_backward_produces_finite_input_and_parameter_gradients(self) -> None:
        module = DGFM(dims=8, out_channels=12)
        rgb = torch.randn(2, 3, 4, 8, requires_grad=True)
        dsm = torch.randn(2, 3, 4, 8, requires_grad=True)

        outputs = module(rgb, dsm)
        loss = sum(output.square().mean() for output in outputs)
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
