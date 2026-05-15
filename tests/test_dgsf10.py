from __future__ import annotations

import unittest

import torch

from models.mfnet.modules import DGSF10, DepthGuidedScaleFusion10
from utils import IntermediateStatsRecorder


class DepthGuidedScaleFusion10Test(unittest.TestCase):
    def _make_dual_features(
        self,
        channels: tuple[int, int, int, int, int] = (8, 8, 8, 8, 8),
        *,
        requires_grad: bool = False,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        rgb_feats = tuple(torch.randn(2, channel, 4, 6, requires_grad=requires_grad) for channel in channels)
        aux_feats = tuple(torch.randn(2, channel, 4, 6, requires_grad=requires_grad) for channel in channels)
        return rgb_feats, aux_feats

    def test_forward_returns_four_projected_multiscale_features(self) -> None:
        channels = (3, 5, 7, 11, 13)
        module = DGSF10(input_channels=channels, hidden_channels=8)
        rgb_feats, aux_feats = self._make_dual_features(channels)

        output = module(rgb_feats, aux_feats)

        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 4)
        for actual, expected_size in zip(output, [(16, 24), (8, 12), (4, 6), (2, 3)]):
            self.assertEqual(actual.shape, (2, 8, *expected_size))
            self.assertTrue(torch.isfinite(actual).all())

    def test_level_fuse_layers_keep_per_level_channels_before_projection(self) -> None:
        module = DGSF10(input_channels=(3, 5, 7, 11, 13), hidden_channels=8)

        self.assertEqual(module.level_fuse1.fuse[0].in_channels, 6)
        self.assertEqual(module.level_fuse1.fuse[0].out_channels, 3)
        self.assertEqual(module.level_fuse2.fuse[0].in_channels, 10)
        self.assertEqual(module.level_fuse2.fuse[0].out_channels, 5)
        self.assertEqual(module.level_fuse3.fuse[0].in_channels, 14)
        self.assertEqual(module.level_fuse3.fuse[0].out_channels, 7)
        self.assertEqual(module.level_fuse4.fuse[0].in_channels, 22)
        self.assertEqual(module.level_fuse4.fuse[0].out_channels, 11)
        self.assertEqual(module.level_fuse_top.fuse[0].in_channels, 26)
        self.assertEqual(module.level_fuse_top.fuse[0].out_channels, 13)

    def test_initializes_residual_scales_and_depth_weights(self) -> None:
        module = DepthGuidedScaleFusion10(input_channels=16, hidden_channels=16)

        self.assertTrue(torch.allclose(module.gamma.detach(), torch.full((4,), 1e-3)))
        self.assertTrue(torch.allclose(module.delta.detach(), torch.full((4,), 1e-3)))
        self.assertTrue(torch.allclose(torch.softmax(module.depth_logits.detach(), dim=0), torch.full((5,), 0.2)))

    def test_rejects_invalid_inputs(self) -> None:
        module = DGSF10(input_channels=8, hidden_channels=8)
        rgb_feats = tuple(torch.randn(2, 8, 4, 4) for _ in range(5))
        aux_feats = tuple(torch.randn(2, 8, 4, 4) for _ in range(5))

        with self.assertRaises(ValueError):
            module(rgb_feats[:4], aux_feats)
        with self.assertRaises(TypeError):
            module(tuple(object() for _ in range(5)), aux_feats)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            module((rgb_feats[0][:, 0], *rgb_feats[1:]), aux_feats)
        with self.assertRaises(ValueError):
            module((torch.randn(2, 4, 4, 4), *rgb_feats[1:]), aux_feats)
        with self.assertRaises(ValueError):
            module((rgb_feats[0], torch.randn(3, 8, 4, 4), *rgb_feats[2:]), aux_feats)
        with self.assertRaises(ValueError):
            module((rgb_feats[0], torch.randn(2, 8, 2, 4), *rgb_feats[2:]), aux_feats)
        with self.assertRaises(ValueError):
            module(rgb_feats, (torch.randn(2, 8, 2, 4), *aux_feats[1:]))

    def test_backward_produces_finite_input_and_parameter_gradients(self) -> None:
        module = DGSF10(input_channels=8, hidden_channels=8)
        rgb_feats, aux_feats = self._make_dual_features(requires_grad=True)

        outputs = module(rgb_feats, aux_feats)
        loss = sum(output.square().mean() for output in outputs)
        loss.backward()

        for feature in (*rgb_feats, *aux_feats):
            self.assertIsNotNone(feature.grad)
            self.assertTrue(torch.isfinite(feature.grad).all())

        trainable_parameters = [param for param in module.parameters() if param.requires_grad]
        self.assertGreater(len(trainable_parameters), 0)
        for param in trainable_parameters:
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())

    def test_zero_delta_returns_scale_branch_main_paths(self) -> None:
        torch.manual_seed(0)
        module = DGSF10(input_channels=8, hidden_channels=8)
        module.delta.data.zero_()
        rgb_feats, aux_feats = self._make_dual_features()

        output = module(rgb_feats, aux_feats)

        with torch.no_grad():
            f1 = module.level_fuse1(rgb_feats[0], aux_feats[0])
            f2 = module.level_fuse2(rgb_feats[1], aux_feats[1])
            f3 = module.level_fuse3(rgb_feats[2], aux_feats[2])
            f4 = module.level_fuse4(rgb_feats[3], aux_feats[3])
            ftop = module.level_fuse_top(rgb_feats[4], aux_feats[4])
            p1 = module.proj1(f1)
            p2 = module.proj2(f2)
            p3 = module.proj3(f3)
            p4 = module.proj4(f4)
            top = module.proj_top(ftop)
            h4, _, _ = module._top_down_step(p4, top, module.gate_fuse4, module.gamma[3])
            h3, _, _ = module._top_down_step(p3, h4, module.gate_fuse3, module.gamma[2])
            h2, _, _ = module._top_down_step(p2, h3, module.gate_fuse2, module.gamma[1])
            h1, _, _ = module._top_down_step(p1, h2, module.gate_fuse1, module.gamma[0])
            base_size = tuple(h1.shape[-2:])
            a1 = module.aggregate1(h1, base_size, align_corners=module.align_corners)
            a2 = module.aggregate2(h2, base_size, align_corners=module.align_corners)
            a3 = module.aggregate3(h3, base_size, align_corners=module.align_corners)
            a4 = module.aggregate4(h4, base_size, align_corners=module.align_corners)
            at = module.aggregate_top(top, base_size, align_corners=module.align_corners)
            weights = torch.softmax(module.depth_logits, dim=0)
            shared = weights[0] * a1 + weights[1] * a2 + weights[2] * a3 + weights[3] * a4 + weights[4] * at
            expected = (
                module.scale_branch1(shared, align_corners=module.align_corners),
                module.scale_branch2(shared, align_corners=module.align_corners),
                module.scale_branch3(shared, align_corners=module.align_corners),
                module.scale_branch4(shared, align_corners=module.align_corners),
            )

        for actual, expected_scale in zip(output, expected):
            self.assertTrue(torch.allclose(actual, expected_scale))

    def test_records_optional_debug_stats(self) -> None:
        module = DGSF10(input_channels=8, hidden_channels=8)
        module.intermediate_stats = IntermediateStatsRecorder()
        module.intermediate_stats_prefix = "dgsf10"
        rgb_feats = tuple(torch.randn(2, 8, 4, 4) for _ in range(5))
        aux_feats = tuple(torch.randn(2, 8, 4, 4) for _ in range(5))

        module(rgb_feats, aux_feats)
        stats = module.intermediate_stats.snapshot()

        expected_keys = {
            "dgsf10/level_fuse/f1_mean",
            "dgsf10/level_fuse/f1_std",
            "dgsf10/level_fuse/f1_over_rgb",
            "dgsf10/level_fuse/f1_over_aux",
            "dgsf10/level_fuse/f2_mean",
            "dgsf10/level_fuse/f2_std",
            "dgsf10/level_fuse/f2_over_rgb",
            "dgsf10/level_fuse/f2_over_aux",
            "dgsf10/level_fuse/f3_mean",
            "dgsf10/level_fuse/f3_std",
            "dgsf10/level_fuse/f3_over_rgb",
            "dgsf10/level_fuse/f3_over_aux",
            "dgsf10/level_fuse/f4_mean",
            "dgsf10/level_fuse/f4_std",
            "dgsf10/level_fuse/f4_over_rgb",
            "dgsf10/level_fuse/f4_over_aux",
            "dgsf10/level_fuse/ftop_mean",
            "dgsf10/level_fuse/ftop_std",
            "dgsf10/level_fuse/ftop_over_rgb",
            "dgsf10/level_fuse/ftop_over_aux",
            "dgsf10/gate/g1_mean",
            "dgsf10/gate/g1_std",
            "dgsf10/gate/g2_mean",
            "dgsf10/gate/g2_std",
            "dgsf10/gate/g3_mean",
            "dgsf10/gate/g3_std",
            "dgsf10/gate/g4_mean",
            "dgsf10/gate/g4_std",
            "dgsf10/residual/gamma1",
            "dgsf10/residual/gamma2",
            "dgsf10/residual/gamma3",
            "dgsf10/residual/gamma4",
            "dgsf10/residual/delta1",
            "dgsf10/residual/delta2",
            "dgsf10/residual/delta3",
            "dgsf10/residual/delta4",
            "dgsf10/depth_weight/w1",
            "dgsf10/depth_weight/w2",
            "dgsf10/depth_weight/w3",
            "dgsf10/depth_weight/w4",
            "dgsf10/depth_weight/wt",
            "dgsf10/feature_ratio/topdown1_over_p1",
            "dgsf10/feature_ratio/topdown2_over_p2",
            "dgsf10/feature_ratio/topdown3_over_p3",
            "dgsf10/feature_ratio/topdown4_over_p4",
            "dgsf10/feature_ratio/fuse1_over_h1",
            "dgsf10/feature_ratio/fuse2_over_h2",
            "dgsf10/feature_ratio/fuse3_over_h3",
            "dgsf10/feature_ratio/fuse4_over_h4",
        }
        self.assertTrue(expected_keys.issubset(stats.keys()))
        for key in expected_keys:
            self.assertIsInstance(stats[key], float)


if __name__ == "__main__":
    unittest.main()
