from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


def _load_dga_module() -> object:
    dga_path = Path(__file__).resolve().parents[1] / "models" / "mfnet" / "modules" / "dga.py"
    spec = importlib.util.spec_from_file_location("_dga_under_test", dga_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load DGA module from {dga_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_dga2_module() -> object:
    dga2_path = Path(__file__).resolve().parents[1] / "models" / "mfnet" / "modules" / "dga2.py"
    spec = importlib.util.spec_from_file_location("_dga2_under_test", dga2_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load DGA2 module from {dga2_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dga = _load_dga_module()
dga2 = _load_dga2_module()


class DGABlockTest(unittest.TestCase):
    def test_forward_returns_pair_with_unchanged_shapes(self) -> None:
        block = dga.DGABlock(channels=16)
        rgb = torch.randn(2, 16, 7, 11)
        aux = torch.randn(2, 16, 7, 11)

        output = block(rgb, aux)

        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 2)
        rgb_out, aux_out = output
        self.assertEqual(rgb_out.shape, rgb.shape)
        self.assertEqual(aux_out.shape, aux.shape)

    def test_rejects_invalid_inputs(self) -> None:
        block = dga.DGABlock(channels=8)
        rgb = torch.randn(2, 8, 6, 6)
        aux = torch.randn(2, 8, 6, 6)

        with self.assertRaises(ValueError):
            block(rgb[:, 0], aux)
        with self.assertRaises(ValueError):
            block(rgb, aux[:, :, :5, :])
        with self.assertRaises(ValueError):
            block(torch.randn(2, 4, 6, 6), aux)

    def test_backward_produces_finite_input_and_parameter_gradients(self) -> None:
        block = dga.DGABlock(channels=8, reduction=4)
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


class DGABlockV2Test(unittest.TestCase):
    def test_forward_returns_pair_with_unchanged_shapes(self) -> None:
        block = dga2.DGABlockV2(channels=16)
        x = torch.randn(2, 16, 7, 11)
        y = torch.randn(2, 16, 7, 11)

        output = block(x, y)

        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 2)
        x_out, y_out = output
        self.assertEqual(x_out.shape, x.shape)
        self.assertEqual(y_out.shape, y.shape)

    def test_rejects_invalid_inputs(self) -> None:
        block = dga2.DGABlockV2(channels=8)
        x = torch.randn(2, 8, 6, 6)
        y = torch.randn(2, 8, 6, 6)

        with self.assertRaises(ValueError):
            block(x[:, 0], y)
        with self.assertRaises(ValueError):
            block(x, y[:, :, :5, :])
        with self.assertRaises(ValueError):
            block(torch.randn(2, 4, 6, 6), y)

    def test_backward_produces_finite_input_and_parameter_gradients(self) -> None:
        block = dga2.DGABlockV2(channels=8)
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
        norm = dga2.LayerNorm2d(3)
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
        block = dga2.DGABlockV2(channels=64)

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
                block = dga2.DGABlockV2(channels=channels)
                self.assertEqual(block.hidden_channels, hidden_channels)
                self.assertEqual(block.message_y_to_x.depthwise.groups, hidden_channels)
                self.assertEqual(block.message_x_to_y.depthwise.groups, hidden_channels)
                self.assertEqual(block.gate_x.depthwise.groups, channels)
                self.assertEqual(block.gate_y.depthwise.groups, channels)
                self.assertEqual(block.message_y_to_x.se.fc1.out_channels, max(8, hidden_channels // 4))
                self.assertEqual(block.message_x_to_y.se.fc1.out_channels, max(8, hidden_channels // 4))

    def test_gate_final_bias_is_zero(self) -> None:
        block = dga2.DGABlockV2(channels=64)

        self.assertTrue(torch.equal(block.gate_x.proj_out.bias.detach(), torch.zeros_like(block.gate_x.proj_out.bias)))
        self.assertTrue(torch.equal(block.gate_y.proj_out.bias.detach(), torch.zeros_like(block.gate_y.proj_out.bias)))

    def test_message_branch_ends_with_linear_projection(self) -> None:
        block = dga2.DGABlockV2(channels=64)

        for message_branch in [block.message_y_to_x, block.message_x_to_y]:
            children = list(message_branch.children())
            self.assertIs(children[-1], message_branch.proj_out)
            self.assertIsInstance(children[-1], torch.nn.Conv2d)
            self.assertNotIsInstance(children[-1], (torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.ReLU, torch.nn.GELU))

    def test_directional_branches_do_not_share_modules(self) -> None:
        block = dga2.DGABlockV2(channels=64)

        self.assertIsNot(block.norm_x, block.norm_y)
        self.assertIsNot(block.message_y_to_x, block.message_x_to_y)
        self.assertIsNot(block.gate_x, block.gate_y)


if __name__ == "__main__":
    unittest.main()
