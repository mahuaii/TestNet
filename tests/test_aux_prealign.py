from __future__ import annotations

import importlib.util
import unittest

import torch

from models.mfnet.aux_prealign import AuxPreAlign

if importlib.util.find_spec("timm") is not None:
    from models.mfnet.UNetFormer_MMSAM_prealign import UNetFormerPreAlign
else:
    UNetFormerPreAlign = None


class _SpyAuxPreAlign(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_shape = tuple(x.shape)
        return x.repeat(1, 3, 1, 1)


class _SpyImageEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_aux_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_aux_shape = tuple(y.shape)
        feat = torch.zeros(x.shape[0], 256, max(1, x.shape[-2] // 16), max(1, x.shape[-1] // 16))
        return feat, feat


class _Fusion(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del y
        return x


class _Decoder(torch.nn.Module):
    def forward(
        self,
        res1: torch.Tensor,
        res2: torch.Tensor,
        res3: torch.Tensor,
        res4: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        del res2, res3, res4
        return torch.zeros(res1.shape[0], 2, h, w)


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


class UNetFormerPreAlignTest(unittest.TestCase):
    @unittest.skipIf(UNetFormerPreAlign is None, "timm is required to import UNetFormerPreAlign")
    def test_forward_adds_channel_dimension_to_batched_dsm(self) -> None:
        assert UNetFormerPreAlign is not None
        model = UNetFormerPreAlign.__new__(UNetFormerPreAlign)
        torch.nn.Module.__init__(model)
        model.aux_prealign = _SpyAuxPreAlign()
        model.image_encoder = _SpyImageEncoder()
        model.fpn1x = torch.nn.Identity()
        model.fpn2x = torch.nn.Identity()
        model.fpn3x = torch.nn.Identity()
        model.fpn4x = torch.nn.Identity()
        model.fpn1y = torch.nn.Identity()
        model.fpn2y = torch.nn.Identity()
        model.fpn3y = torch.nn.Identity()
        model.fpn4y = torch.nn.Identity()
        model.fusion1 = _Fusion()
        model.fusion2 = _Fusion()
        model.fusion3 = _Fusion()
        model.fusion4 = _Fusion()
        model.decoder = _Decoder()

        output = model(torch.randn(2, 3, 64, 64), torch.randn(2, 64, 64))

        self.assertEqual(output.shape, (2, 2, 64, 64))
        self.assertEqual(model.aux_prealign.last_shape, (2, 1, 64, 64))
        self.assertEqual(model.image_encoder.last_aux_shape, (2, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
