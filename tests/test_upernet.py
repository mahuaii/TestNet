from __future__ import annotations

import unittest

import torch

from models.mfnet.modules import UperNetHead


class UperNetHeadTest(unittest.TestCase):
    def test_forward_accepts_four_multiscale_features(self) -> None:
        head = UperNetHead(in_channels=(256, 256, 256, 256), channels=32, num_classes=6)
        head.eval()
        feats = [
            torch.randn(2, 256, 16, 16),
            torch.randn(2, 256, 8, 8),
            torch.randn(2, 256, 4, 4),
            torch.randn(2, 256, 2, 2),
        ]

        with torch.no_grad():
            output = head(feats, 64, 64)

        self.assertEqual(output.shape, (2, 6, 64, 64))
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_accepts_four_same_scale_features(self) -> None:
        head = UperNetHead(in_channels=(256, 256, 256, 256), channels=32, num_classes=6)
        head.eval()
        feats = [torch.randn(2, 256, 8, 8) for _ in range(4)]

        with torch.no_grad():
            output = head(feats, 32, 48)

        self.assertEqual(output.shape, (2, 6, 32, 48))
        self.assertTrue(torch.isfinite(output).all())

    def test_rejects_wrong_feature_count(self) -> None:
        head = UperNetHead(in_channels=(256, 256, 256, 256), channels=32, num_classes=6)

        with self.assertRaises(ValueError):
            head([torch.randn(2, 256, 8, 8) for _ in range(3)], 32, 32)

    def test_rejects_wrong_input_channel_config_count(self) -> None:
        with self.assertRaises(ValueError):
            UperNetHead(in_channels=(256, 256, 256), channels=32, num_classes=6)


if __name__ == "__main__":
    unittest.main()
