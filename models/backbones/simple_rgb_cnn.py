from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleRGBBackbone(nn.Module):
    """Minimal RGB backbone for pipeline validation."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBlock(3, 16)
        self.stage1 = ConvBlock(16, 32, stride=2)
        self.stage2 = ConvBlock(32, 64, stride=2)

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        feat0 = self.stem(x)
        feat1 = self.stage1(feat0)
        feat2 = self.stage2(feat1)
        return {
            "rgb_feats": [feat0, feat1, feat2],
            "fused_feats": [feat2],
        }
