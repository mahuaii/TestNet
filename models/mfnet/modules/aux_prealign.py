from __future__ import annotations

import torch
import torch.nn as nn


class AuxPreAlign(nn.Module):
    """Map a single-channel auxiliary tensor to a 3-channel SAM-compatible feature map."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        branch_channels: int = 32,
        fusion_channels: int = 32,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stem = self._make_conv_block(
            in_ch=in_channels,
            out_ch=base_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
        )
        self.branch1 = self._make_conv_block(
            in_ch=base_channels,
            out_ch=branch_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
        )
        self.branch2 = self._make_conv_block(
            in_ch=base_channels,
            out_ch=branch_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
        )
        self.branch3 = self._make_conv_block(
            in_ch=base_channels,
            out_ch=branch_channels,
            kernel_size=3,
            padding=3,
            dilation=3,
        )
        self.fuse = self._make_conv_block(
            in_ch=branch_channels * 3,
            out_ch=fusion_channels,
            kernel_size=1,
            padding=0,
            dilation=1,
        )
        self.project = nn.Conv2d(
            in_channels=fusion_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    @staticmethod
    def _make_conv_block(
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        padding: int,
        dilation: int = 1,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor of shape (B, C, H, W), but got shape {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected input with {self.in_channels} channel(s), but got {x.shape[1]} channel(s).")

        height, width = x.shape[-2:]

        feat = self.stem(x)
        multi_scale = torch.cat((self.branch1(feat), self.branch2(feat), self.branch3(feat)), dim=1)
        fused = self.fuse(multi_scale)
        y = self.project(fused)

        residual = x.repeat(1, self.out_channels, 1, 1)
        y = y + residual

        if y.shape[-2:] != (height, width):
            raise RuntimeError(
                f"Spatial size changed unexpectedly: input {(height, width)}, output {tuple(y.shape[-2:])}."
            )

        self._record_intermediate_stats(y)
        return y

    def _record_intermediate_stats(self, output: torch.Tensor) -> None:
        stats = getattr(self, "intermediate_stats", None)
        if stats is None:
            return
        prefix = str(getattr(self, "intermediate_stats_prefix", "prealign")).strip("/")
        tensor = output.detach()
        stats.record_scalar(f"{prefix}/output_mean", tensor.mean())
        stats.record_scalar(f"{prefix}/output_std", tensor.std(unbiased=False))
        stats.record_scalar(f"{prefix}/output_var", tensor.var(unbiased=False))
        stats.record_norm(f"{prefix}/output_norm", tensor)
