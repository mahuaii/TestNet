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


dga = _load_dga_module()


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


if __name__ == "__main__":
    unittest.main()
