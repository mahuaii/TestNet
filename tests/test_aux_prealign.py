from __future__ import annotations

import unittest

import torch

from models.mfnet.aux_prealign import AuxPreAlign


class AuxPreAlignTest(unittest.TestCase):
    def test_forward_returns_three_channel_output_with_same_spatial_size(self) -> None:
        model = AuxPreAlign()
        x = torch.randn(2, 1, 128, 128)

        y = model(x)

        self.assertEqual(y.shape, (2, 3, 128, 128))

    def test_forward_rejects_unexpected_channel_count(self) -> None:
        model = AuxPreAlign()
        x = torch.randn(2, 2, 64, 64)

        with self.assertRaisesRegex(ValueError, "Expected input with 1 channel"):
            model(x)

    def test_forward_rejects_non_4d_input(self) -> None:
        model = AuxPreAlign()
        x = torch.randn(1, 64, 64)

        with self.assertRaisesRegex(ValueError, "Expected a 4D tensor"):
            model(x)


if __name__ == "__main__":
    unittest.main()
