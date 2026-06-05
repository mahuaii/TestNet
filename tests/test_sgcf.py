from __future__ import annotations

import unittest

import torch

from models.mfnet.modules import MultiScaleSGCF, SGCF, SGCFBlock, SGCFScaleAdapter, SobelDSMEdge


class SGCFBlockTest(unittest.TestCase):
    def test_forward_returns_output_and_spatial_mask(self) -> None:
        module = SGCFBlock(in_channels=8, hidden_dim=16, groups=8)
        rgb = torch.randn(2, 8, 5, 7)
        aux = torch.randn(2, 8, 5, 7)
        dsm_edge = torch.randn(2, 1, 20, 28)

        output, mask = module(rgb, aux, dsm_edge)

        self.assertEqual(output.shape, rgb.shape)
        self.assertEqual(mask.shape, (2, 1, 5, 7))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(mask).all())
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0)

    def test_initializes_gamma(self) -> None:
        module = SGCFBlock(in_channels=8, hidden_dim=16, groups=8)

        self.assertTrue(torch.allclose(module.gamma.detach(), torch.full((1,), 1e-3)))

    def test_zero_gamma_returns_rgb_input(self) -> None:
        torch.manual_seed(0)
        module = SGCFBlock(in_channels=8, hidden_dim=16, groups=8)
        module.gamma.data.zero_()
        rgb = torch.randn(2, 8, 5, 7)
        aux = torch.randn(2, 8, 5, 7)
        dsm_edge = torch.randn(2, 1, 5, 7)

        output, _ = module(rgb, aux, dsm_edge)

        self.assertTrue(torch.equal(output, rgb))

    def test_aux_channel_mismatch_uses_projection(self) -> None:
        module = SGCFBlock(in_channels=8, aux_channels=5, hidden_dim=16, groups=8)
        rgb = torch.randn(2, 8, 5, 7)
        aux = torch.randn(2, 5, 5, 7)
        dsm_edge = torch.randn(2, 1, 5, 7)

        output, mask = module(rgb, aux, dsm_edge)

        self.assertIsInstance(module.aux_match, torch.nn.Conv2d)
        self.assertEqual(module.aux_match.in_channels, 5)
        self.assertEqual(module.aux_match.out_channels, 8)
        self.assertEqual(output.shape, rgb.shape)
        self.assertEqual(mask.shape, (2, 1, 5, 7))

    def test_rejects_invalid_inputs(self) -> None:
        module = SGCFBlock(in_channels=8, hidden_dim=16, groups=8)
        rgb = torch.randn(2, 8, 5, 7)
        aux = torch.randn(2, 8, 5, 7)
        dsm_edge = torch.randn(2, 1, 5, 7)

        with self.assertRaises(ValueError):
            module(rgb[:, 0], aux, dsm_edge)
        with self.assertRaises(ValueError):
            module(torch.randn(2, 4, 5, 7), aux, dsm_edge)
        with self.assertRaises(ValueError):
            module(rgb, torch.randn(2, 4, 5, 7), dsm_edge)
        with self.assertRaises(ValueError):
            module(rgb, torch.randn(2, 8, 4, 7), dsm_edge)
        with self.assertRaises(ValueError):
            module(rgb, aux, dsm_edge[:, 0])
        with self.assertRaises(ValueError):
            module(rgb, aux, torch.randn(2, 3, 5, 7))
        with self.assertRaises(ValueError):
            module(rgb, aux, torch.randn(3, 1, 5, 7))

    def test_rejects_invalid_constructor_values(self) -> None:
        with self.assertRaises(ValueError):
            SGCFBlock(in_channels=0)
        with self.assertRaises(ValueError):
            SGCFBlock(in_channels=8, hidden_dim=10, groups=8)

    def test_backward_produces_finite_input_and_parameter_gradients(self) -> None:
        module = SGCFBlock(in_channels=8, aux_channels=5, hidden_dim=16, groups=8)
        rgb = torch.randn(2, 8, 5, 7, requires_grad=True)
        aux = torch.randn(2, 5, 5, 7, requires_grad=True)
        dsm_edge = torch.randn(2, 1, 10, 14, requires_grad=True)

        output, mask = module(rgb, aux, dsm_edge)
        loss = output.square().mean() + mask.square().mean()
        loss.backward()

        for feature in (rgb, aux, dsm_edge):
            self.assertIsNotNone(feature.grad)
            self.assertTrue(torch.isfinite(feature.grad).all())

        trainable_parameters = [param for param in module.parameters() if param.requires_grad]
        self.assertGreater(len(trainable_parameters), 0)
        for param in trainable_parameters:
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())


class SGCFTest(unittest.TestCase):
    def test_forward_accepts_bhwc_and_returns_decoder_features(self) -> None:
        module = SGCF(dims=8, hidden_dim=16, out_channels=12, groups=8)
        rgb = torch.randn(2, 4, 6, 8)
        aux = torch.randn(2, 4, 6, 8)
        dsm = torch.randn(2, 1, 16, 24)

        outputs = module(rgb, aux, dsm)

        self.assertEqual(len(outputs), 4)
        for output, expected_size in zip(outputs, [(16, 24), (8, 12), (4, 6), (2, 3)]):
            self.assertEqual(output.shape, (2, 12, *expected_size))
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(module.last_spatial_mask.shape, (2, 1, 4, 6))

    def test_zero_gamma_returns_projected_rgb_path(self) -> None:
        torch.manual_seed(0)
        module = SGCF(dims=8, hidden_dim=16, out_channels=12, groups=8)
        module.fusion.gamma.data.zero_()
        rgb = torch.randn(2, 4, 6, 8)
        aux = torch.randn(2, 4, 6, 8)
        dsm = torch.randn(2, 1, 16, 24)

        outputs = module(rgb, aux, dsm)

        with torch.no_grad():
            expected = module.input_norm(rgb).permute(0, 3, 1, 2).contiguous()
            expected = module.output_norm(expected.permute(0, 2, 3, 1))
            expected = expected.permute(0, 3, 1, 2).contiguous()
            expected = module.output_proj(expected)
            expected_outputs = module.scale_adapter(expected)

        for output, expected_output in zip(outputs, expected_outputs):
            self.assertTrue(torch.allclose(output, expected_output))

    def test_backward_produces_finite_input_and_parameter_gradients(self) -> None:
        module = SGCF(dims=8, hidden_dim=16, out_channels=12, groups=8)
        rgb = torch.randn(2, 4, 6, 8, requires_grad=True)
        aux = torch.randn(2, 4, 6, 8, requires_grad=True)
        dsm = torch.randn(2, 1, 16, 24, requires_grad=True)

        outputs = module(rgb, aux, dsm)
        loss = sum(output.square().mean() for output in outputs)
        loss.backward()

        for feature in (rgb, aux, dsm):
            self.assertIsNotNone(feature.grad)
            self.assertTrue(torch.isfinite(feature.grad).all())

        trainable_parameters = [param for param in module.parameters() if param.requires_grad]
        self.assertGreater(len(trainable_parameters), 0)
        for param in trainable_parameters:
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())

    def test_rejects_invalid_bhwc_inputs(self) -> None:
        module = SGCF(dims=8, hidden_dim=16, out_channels=12, groups=8)
        rgb = torch.randn(2, 4, 6, 8)
        aux = torch.randn(2, 4, 6, 8)
        dsm = torch.randn(2, 1, 16, 24)

        with self.assertRaises(ValueError):
            module(rgb.permute(0, 3, 1, 2), aux, dsm)
        with self.assertRaises(ValueError):
            module(rgb, torch.randn(2, 4, 6, 4), dsm)
        with self.assertRaises(ValueError):
            module(rgb, torch.randn(2, 4, 5, 8), dsm)


class SGCFScaleAdapterTest(unittest.TestCase):
    def test_adapts_four_independent_fused_features_to_decoder_scales(self) -> None:
        adapter = SGCFScaleAdapter(channels=12)
        features = tuple(torch.randn(2, 12, 4, 6) for _ in range(4))

        outputs = adapter(features)

        self.assertEqual(len(outputs), 4)
        for actual, expected_size in zip(outputs, [(16, 24), (8, 12), (4, 6), (2, 3)]):
            self.assertEqual(actual.shape, (2, 12, *expected_size))
            self.assertTrue(torch.isfinite(actual).all())

    def test_rejects_invalid_features(self) -> None:
        adapter = SGCFScaleAdapter(channels=12)
        features = tuple(torch.randn(2, 12, 4, 6) for _ in range(4))

        with self.assertRaises(ValueError):
            adapter(features[:3])
        with self.assertRaises(ValueError):
            adapter((torch.randn(2, 8, 4, 6), *features[1:]))


class SobelDSMEdgeTest(unittest.TestCase):
    def test_accepts_three_or_four_dimensional_single_channel_dsm(self) -> None:
        edge = SobelDSMEdge()
        dsm = torch.randn(2, 8, 10)

        edge_3d = edge(dsm)
        edge_4d = edge(dsm.unsqueeze(1))

        self.assertEqual(edge_3d.shape, (2, 1, 8, 10))
        self.assertTrue(torch.allclose(edge_3d, edge_4d))
        self.assertTrue(torch.isfinite(edge_3d).all())


class MultiScaleSGCFTest(unittest.TestCase):
    def test_forward_returns_four_outputs_and_masks(self) -> None:
        channels = (8, 12, 16, 20)
        module = MultiScaleSGCF(channels=channels, hidden_dim=16, groups=8)
        sizes = ((16, 20), (8, 10), (4, 5), (2, 3))
        rgb_feats = tuple(torch.randn(2, channel, *size) for channel, size in zip(channels, sizes))
        aux_feats = tuple(torch.randn(2, channel, *size) for channel, size in zip(channels, sizes))
        dsm_edge = torch.randn(2, 1, 32, 40)

        outputs, masks = module(rgb_feats, aux_feats, dsm_edge)

        self.assertIsInstance(outputs, tuple)
        self.assertIsInstance(masks, tuple)
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(masks), 4)
        for output, mask, channel, size in zip(outputs, masks, channels, sizes):
            self.assertEqual(output.shape, (2, channel, *size))
            self.assertEqual(mask.shape, (2, 1, *size))
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.isfinite(mask).all())

    def test_supports_per_scale_aux_channels(self) -> None:
        module = MultiScaleSGCF(
            channels=(8, 8, 8, 8),
            aux_channels=(3, 4, 5, 6),
            hidden_dim=16,
            groups=8,
        )
        rgb_feats = tuple(torch.randn(2, 8, 4, 4) for _ in range(4))
        aux_feats = tuple(torch.randn(2, channel, 4, 4) for channel in (3, 4, 5, 6))
        dsm_edge = torch.randn(2, 1, 4, 4)

        outputs, masks = module(rgb_feats, aux_feats, dsm_edge)

        self.assertEqual([output.shape for output in outputs], [(2, 8, 4, 4)] * 4)
        self.assertEqual([mask.shape for mask in masks], [(2, 1, 4, 4)] * 4)

    def test_rejects_invalid_feature_sequence_lengths(self) -> None:
        module = MultiScaleSGCF(channels=(8, 8, 8, 8), hidden_dim=16, groups=8)
        rgb_feats = tuple(torch.randn(2, 8, 4, 4) for _ in range(4))
        aux_feats = tuple(torch.randn(2, 8, 4, 4) for _ in range(4))
        dsm_edge = torch.randn(2, 1, 4, 4)

        with self.assertRaises(ValueError):
            module(rgb_feats[:3], aux_feats, dsm_edge)
        with self.assertRaises(ValueError):
            module(rgb_feats, aux_feats[:3], dsm_edge)
        with self.assertRaises(TypeError):
            module(object(), aux_feats, dsm_edge)  # type: ignore[arg-type]

    def test_rejects_invalid_constructor_channel_lengths(self) -> None:
        with self.assertRaises(ValueError):
            MultiScaleSGCF(channels=(8, 8, 8), hidden_dim=16, groups=8)
        with self.assertRaises(ValueError):
            MultiScaleSGCF(channels=(8, 8, 8, 8), aux_channels=(8, 8, 8), hidden_dim=16, groups=8)


if __name__ == "__main__":
    unittest.main()
