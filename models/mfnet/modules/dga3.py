from __future__ import annotations

import torch
import torch.nn as nn

from .dga2 import LayerNorm2d


def _validate_pair(x: torch.Tensor, y: torch.Tensor, channels: int) -> None:
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError("Expected x and y to be 4D tensors with shape [B, C, H, W].")
    if x.shape != y.shape:
        raise ValueError(f"Expected x and y to have the same shape, got {tuple(x.shape)} and {tuple(y.shape)}.")
    if x.shape[1] != channels:
        raise ValueError(f"Expected input channel count {channels}, got {x.shape[1]}.")


class _SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        return x * self.gate(scale)


class _GateBranch(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj_in = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.act = nn.GELU()
        self.se = _SqueezeExcitation(channels)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.act(x)
        x = self.se(x)
        x = self.proj_out(x)
        return self.gate(x)


class DGABlockV3(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"Expected channels to be positive, got {channels}.")
        self.channels = int(channels)

        self.norm_x = LayerNorm2d(channels)
        self.norm_y = LayerNorm2d(channels)
        self.gate_x = _GateBranch(channels)
        self.gate_y = _GateBranch(channels)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_pair(x, y, self.channels)
        x_norm = self.norm_x(x)
        y_norm = self.norm_y(y)
        difference = x_norm - y_norm

        gate_x = self.gate_x(torch.cat([x_norm, difference], dim=1))
        gate_y = self.gate_y(torch.cat([y_norm, -difference], dim=1))

        x_out = x + gate_x * y_norm
        y_out = y + gate_y * x_norm
        return x_out, y_out


__all__ = ["DGABlockV3"]
