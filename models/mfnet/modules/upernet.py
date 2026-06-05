from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        bias: bool = False,
    ) -> None:
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=padding,
            ),
            norm_layer(out_channels),
            nn.ReLU6(),
        )


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        bias: bool = False,
    ) -> None:
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=padding,
            ),
            norm_layer(out_channels),
        )


class Conv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = False,
    ) -> None:
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=padding,
            )
        )


class PPM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_sizes: Sequence[int] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    ConvBNReLU(in_channels, out_channels, kernel_size=1),
                )
                for pool_size in pool_sizes
            ]
        )
        self.bottleneck = ConvBNReLU(
            in_channels + len(pool_sizes) * out_channels,
            out_channels,
            kernel_size=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        ppm_outs = [x]
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            ppm_outs.append(y)
        return self.bottleneck(torch.cat(ppm_outs, dim=1))


class UperNetHead(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int,
        num_classes: int,
        dropout: float = 0.1,
        pool_sizes: Sequence[int] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"Expected 4 input channel entries, got {len(in_channels)}.")
        self.ppm = PPM(in_channels[-1], channels, pool_sizes)
        self.lateral_convs = nn.ModuleList(
            [ConvBN(in_channels[index], channels, kernel_size=1) for index in range(len(in_channels) - 1)]
        )
        self.fpn_convs = nn.ModuleList(
            [ConvBNReLU(channels, channels, kernel_size=3) for _ in range(len(in_channels) - 1)]
        )
        self.fpn_bottleneck = ConvBNReLU(
            channels * len(in_channels),
            channels,
            kernel_size=3,
        )
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(channels, num_classes, kernel_size=1),
        )

    def forward(self, feats: Sequence[torch.Tensor], h: int, w: int) -> torch.Tensor:
        if len(feats) != 4:
            raise ValueError(f"Expected 4 feature maps, got {len(feats)}.")
        c1, c2, c3, c4 = feats
        p4 = self.ppm(c4)

        laterals = [
            self.lateral_convs[0](c1),
            self.lateral_convs[1](c2),
            self.lateral_convs[2](c3),
        ]

        p3 = laterals[2] + F.interpolate(p4, size=laterals[2].shape[2:], mode="bilinear", align_corners=False)
        p2 = laterals[1] + F.interpolate(p3, size=laterals[1].shape[2:], mode="bilinear", align_corners=False)
        p1 = laterals[0] + F.interpolate(p2, size=laterals[0].shape[2:], mode="bilinear", align_corners=False)

        fpn_outs = [
            self.fpn_convs[0](p1),
            self.fpn_convs[1](p2),
            self.fpn_convs[2](p3),
            p4,
        ]
        for index in range(1, len(fpn_outs)):
            fpn_outs[index] = F.interpolate(
                fpn_outs[index],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        out = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        out = self.classifier(out)
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


__all__ = ["PPM", "UperNetHead"]
