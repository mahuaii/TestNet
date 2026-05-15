from __future__ import annotations

import unittest

import torch

from models.mfnet.modules import dga10, dga20, dga30, dga_softplus


class DGABlock10Test(unittest.TestCase):
    def test_forward_returns_pair_with_unchanged_shapes(self) -> None:
        block = dga10.DGABlock10(channels=16)
        rgb = torch.randn(2, 16, 7, 11)
        aux = torch.randn(2, 16, 7, 11)

        output = block(rgb, aux)

        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 2)
        rgb_out, aux_out = output
        self.assertEqual(rgb_out.shape, rgb.shape)
        self.assertEqual(aux_out.shape, aux.shape)

    def test_rejects_invalid_inputs(self) -> None:
        block = dga10.DGABlock10(channels=8)
        rgb = torch.randn(2, 8, 6, 6)
        aux = torch.randn(2, 8, 6, 6)

        with self.assertRaises(ValueError):
            block(rgb[:, 0], aux)
        with self.assertRaises(ValueError):
            block(rgb, aux[:, :, :5, :])
        with self.assertRaises(ValueError):
            block(torch.randn(2, 4, 6, 6), aux)

    def test_backward_produces_finite_input_and_parameter_gradients(self) -> None:
        block = dga10.DGABlock10(channels=8, reduction=4)
        rgb = torch.randn(2, 8, 6, 6, requires_grad=True)
        aux = torch.randn(2, 8, 6, 6, requires_grad=True)

        rgb_out, aux_out = block(rgb, aux)
        loss = rgb_out.square().mean() + aux_out.square().mean()
        loss.backward()

        self.assertIsNotNone(rgb.grad)
        self.assertIsNotNone(aux.grad)
        self.assertTrue(torch.isfinite(rgb.grad).all())
        self.assertTrue(torch.isfinite(aux.grad).all())

        trainable_parameters = [param for param in block.parameters() if param.requires_grad]
        self.assertGreater(len(trainable_parameters), 0)
        for param in trainable_parameters:
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())


class DGABlock20Test(unittest.TestCase):
    def test_forward_returns_pair_with_unchanged_shapes(self) -> None:
        block = dga20.DGABlock20(channels=16)
        x = torch.randn(2, 16, 7, 11)
        y = torch.randn(2, 16, 7, 11)

        output = block(x, y)

        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 2)
        x_out, y_out = output
        self.assertEqual(x_out.shape, x.shape)
        self.assertEqual(y_out.shape, y.shape)

    def test_rejects_invalid_inputs(self) -> None:
        block = dga20.DGABlock20(channels=8)
        x = torch.randn(2, 8, 6, 6)
        y = torch.randn(2, 8, 6, 6)

        with self.assertRaises(ValueError):
            block(x[:, 0], y)
        with self.assertRaises(ValueError):
            block(x, y[:, :, :5, :])
        with self.assertRaises(ValueError):
            block(torch.randn(2, 4, 6, 6), y)

    def test_backward_produces_finite_input_and_parameter_gradients(self) -> None:
        block = dga20.DGABlock20(channels=8)
        x = torch.randn(2, 8, 6, 6, requires_grad=True)
        y = torch.randn(2, 8, 6, 6, requires_grad=True)

        x_out, y_out = block(x, y)
        loss = x_out.square().mean() + y_out.square().mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(y.grad).all())

        trainable_parameters = [param for param in block.parameters() if param.requires_grad]
        self.assertGreater(len(trainable_parameters), 0)
        for param in trainable_parameters:
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())

    def test_layer_norm2d_normalizes_channel_dimension_per_spatial_position(self) -> None:
        norm = dga20.LayerNorm2d(3)
        x = torch.tensor(
            [
                [
                    [[1.0, 10.0], [3.0, 5.0]],
                    [[2.0, 20.0], [6.0, 10.0]],
                    [[4.0, 40.0], [9.0, 20.0]],
                ]
            ]
        )

        output = norm(x)

        self.assertTrue(torch.allclose(output.mean(dim=1), torch.zeros(1, 2, 2), atol=1e-5))
        self.assertTrue(torch.allclose(output.var(dim=1, unbiased=False), torch.ones(1, 2, 2), atol=1e-4))

    def test_default_scale_parameters_are_shape_one_and_initialized_to_point_one(self) -> None:
        block = dga20.DGABlock20(channels=64)

        self.assertEqual(tuple(block.alpha.shape), (1,))
        self.assertEqual(tuple(block.beta.shape), (1,))
        self.assertTrue(torch.allclose(block.alpha.detach(), torch.tensor([0.1])))
        self.assertTrue(torch.allclose(block.beta.detach(), torch.tensor([0.1])))

    def test_hidden_channels_follow_dga2_rule(self) -> None:
        expected = {
            64: 32,
            128: 32,
            768: 192,
        }
        for channels, hidden_channels in expected.items():
            with self.subTest(channels=channels):
                block = dga20.DGABlock20(channels=channels)
                self.assertEqual(block.hidden_channels, hidden_channels)
                self.assertEqual(block.message_y_to_x.depthwise.groups, hidden_channels)
                self.assertEqual(block.message_x_to_y.depthwise.groups, hidden_channels)
                self.assertEqual(block.gate_x.depthwise.groups, channels)
                self.assertEqual(block.gate_y.depthwise.groups, channels)
                self.assertEqual(block.message_y_to_x.se.fc1.out_channels, max(8, hidden_channels // 4))
                self.assertEqual(block.message_x_to_y.se.fc1.out_channels, max(8, hidden_channels // 4))

    def test_gate_final_bias_is_zero(self) -> None:
        block = dga20.DGABlock20(channels=64)

        self.assertTrue(torch.equal(block.gate_x.proj_out.bias.detach(), torch.zeros_like(block.gate_x.proj_out.bias)))
        self.assertTrue(torch.equal(block.gate_y.proj_out.bias.detach(), torch.zeros_like(block.gate_y.proj_out.bias)))

    def test_message_branch_ends_with_linear_projection(self) -> None:
        block = dga20.DGABlock20(channels=64)

        for message_branch in [block.message_y_to_x, block.message_x_to_y]:
            children = list(message_branch.children())
            self.assertIs(children[-1], message_branch.proj_out)
            self.assertIsInstance(children[-1], torch.nn.Conv2d)
            self.assertNotIsInstance(children[-1], (torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.ReLU, torch.nn.GELU))

    def test_directional_branches_do_not_share_modules(self) -> None:
        block = dga20.DGABlock20(channels=64)

        self.assertIsNot(block.norm_x, block.norm_y)
        self.assertIsNot(block.message_y_to_x, block.message_x_to_y)
        self.assertIsNot(block.gate_x, block.gate_y)


class DGAIntermediateStatsTest(unittest.TestCase):
    def test_dga10_block_records_layered_intermediate_stats_when_recorder_is_attached(self) -> None:
        from utils import IntermediateStatsRecorder

        block = dga10.DGABlock10(channels=8, reduction=4)
        block.intermediate_stats = IntermediateStatsRecorder()
        block.intermediate_stats_prefix = "dga/block_0"
        rgb = torch.randn(2, 8, 5, 5, requires_grad=True)
        aux = torch.randn(2, 8, 5, 5, requires_grad=True)

        block(rgb, aux)

        stats = block.intermediate_stats.snapshot(reset=True)
        self.assertIn("dga/block_0/rgb_gate_mean", stats)
        self.assertIn("dga/block_0/rgb_gate_std", stats)
        self.assertIn("dga/block_0/aux_gate_mean", stats)
        self.assertIn("dga/block_0/alpha_injection_ratio", stats)
        self.assertIn("dga/block_0/beta_injection_norm", stats)
        for value in stats.values():
            self.assertIsInstance(value, float)
        self.assertEqual(block.intermediate_stats.snapshot(), {})
        self.assertFalse(hasattr(block, "last_dga_stats"))

    def test_dga20_block_records_layered_intermediate_stats_when_recorder_is_attached(self) -> None:
        from utils import IntermediateStatsRecorder

        block = dga20.DGABlock20(channels=8)
        block.intermediate_stats = IntermediateStatsRecorder()
        block.intermediate_stats_prefix = "dga/block_1"
        x = torch.randn(2, 8, 5, 5, requires_grad=True)
        y = torch.randn(2, 8, 5, 5, requires_grad=True)

        block(x, y)

        stats = block.intermediate_stats.snapshot()
        self.assertIn("dga/block_1/x_gate_mean", stats)
        self.assertIn("dga/block_1/x_gate_std", stats)
        self.assertIn("dga/block_1/y_gate_mean", stats)
        self.assertIn("dga/block_1/alpha_injection_ratio", stats)
        self.assertIn("dga/block_1/beta_main_norm", stats)
        for value in stats.values():
            self.assertIsInstance(value, float)
        self.assertFalse(hasattr(block, "last_dga_stats"))

    def test_dga10_softplus_keeps_independent_effective_scale_parameterization(self) -> None:
        block = dga_softplus.DGABlock10Softplus(channels=8, reduction=4, init_scale=0.1)

        self.assertFalse(torch.allclose(block.alpha.detach(), torch.tensor(0.1)))
        self.assertTrue(torch.allclose(block.effective_alpha().detach(), torch.tensor(0.1), atol=1e-6))
        self.assertTrue(torch.allclose(block.effective_beta().detach(), torch.tensor(0.1), atol=1e-6))

    def test_dga20_softplus_keeps_independent_effective_scale_parameterization(self) -> None:
        block = dga_softplus.DGABlock20Softplus(channels=8, init_scale=0.1)

        self.assertFalse(torch.allclose(block.alpha.detach(), torch.tensor([0.1])))
        self.assertTrue(torch.allclose(block.effective_alpha().detach(), torch.tensor([0.1]), atol=1e-6))
        self.assertTrue(torch.allclose(block.effective_beta().detach(), torch.tensor([0.1]), atol=1e-6))


class DGABlock30Test(unittest.TestCase):
    def test_forward_returns_pair_with_unchanged_shapes(self) -> None:
        block = dga30.DGABlock30(channels=16)
        x = torch.randn(2, 16, 7, 11)
        y = torch.randn(2, 16, 7, 11)

        output = block(x, y)

        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 2)
        x_out, y_out = output
        self.assertEqual(x_out.shape, x.shape)
        self.assertEqual(y_out.shape, y.shape)

    def test_rejects_invalid_inputs(self) -> None:
        block = dga30.DGABlock30(channels=8)
        x = torch.randn(2, 8, 6, 6)
        y = torch.randn(2, 8, 6, 6)

        with self.assertRaises(ValueError):
            block(x[:, 0], y)
        with self.assertRaises(ValueError):
            block(x, y[:, :, :5, :])
        with self.assertRaises(ValueError):
            block(torch.randn(2, 4, 6, 6), y)

    def test_backward_produces_finite_input_and_parameter_gradients(self) -> None:
        block = dga30.DGABlock30(channels=8)
        x = torch.randn(2, 8, 6, 6, requires_grad=True)
        y = torch.randn(2, 8, 6, 6, requires_grad=True)

        x_out, y_out = block(x, y)
        loss = x_out.square().mean() + y_out.square().mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(y.grad).all())

        trainable_parameters = [param for param in block.parameters() if param.requires_grad]
        self.assertGreater(len(trainable_parameters), 0)
        for param in trainable_parameters:
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())

    def test_gate_branches_follow_dga30_structure(self) -> None:
        block = dga30.DGABlock30(channels=64)

        for gate_branch in [block.gate_x, block.gate_y]:
            self.assertEqual(gate_branch.proj_in.in_channels, 128)
            self.assertEqual(gate_branch.proj_in.out_channels, 64)
            self.assertIsInstance(gate_branch.act, torch.nn.GELU)
            self.assertIsInstance(gate_branch.se, torch.nn.Module)
            self.assertEqual(gate_branch.proj_out.in_channels, 64)
            self.assertEqual(gate_branch.proj_out.out_channels, 64)
            self.assertIsInstance(gate_branch.gate, torch.nn.Sigmoid)

    def test_uses_normalized_opposite_feature_as_message(self) -> None:
        block = dga30.DGABlock30(channels=4)
        x = torch.randn(2, 4, 5, 5)
        y = torch.randn(2, 4, 5, 5)

        x_norm = block.norm_x(x)
        y_norm = block.norm_y(y)
        difference = x_norm - y_norm
        expected_x = x + block.gate_x(torch.cat([x_norm, difference], dim=1)) * y_norm
        expected_y = y + block.gate_y(torch.cat([y_norm, -difference], dim=1)) * x_norm

        actual_x, actual_y = block(x, y)

        self.assertTrue(torch.allclose(actual_x, expected_x))
        self.assertTrue(torch.allclose(actual_y, expected_y))


if __name__ == "__main__":
    unittest.main()
