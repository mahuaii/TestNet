from __future__ import annotations

import torch
import torch.nn as nn


def _hidden_channels(channels: int) -> int:
    if channels <= 0:
        raise ValueError(f"Expected channels to be positive, got {channels}.")
    return min(channels, max(32, channels // 4))


def _validate_pair(x: torch.Tensor, y: torch.Tensor, channels: int) -> None:
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError("Expected x and y to be 4D tensors with shape [B, C, H, W].")
    if x.shape != y.shape:
        raise ValueError(f"Expected x and y to have the same shape, got {tuple(x.shape)} and {tuple(y.shape)}.")
    if x.shape[1] != channels:
        raise ValueError(f"Expected input channel count {channels}, got {x.shape[1]}.")


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class _SqueezeExcitation(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = max(8, channels // 4)
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


class _MessageBranch(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.proj_in = nn.Conv2d(channels * 2, hidden_channels, kernel_size=1)
        self.act1 = nn.GELU()
        self.depthwise = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.act2 = nn.GELU()
        self.se = _SqueezeExcitation(hidden_channels)
        self.proj_out = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.act1(x)
        x = self.depthwise(x)
        x = self.act2(x)
        x = self.se(x)
        return self.proj_out(x)


class _GateBranch(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj_in = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.act = nn.GELU()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj_out.bias)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.act(x)
        x = self.depthwise(x)
        x = self.proj_out(x)
        return self.gate(x)


class DGABlockV2(nn.Module):
    def __init__(self, channels: int, init_scale: float = 0.1) -> None:
        super().__init__()
        hidden_channels = _hidden_channels(channels)
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.init_scale = float(init_scale)

        self.norm_x = LayerNorm2d(channels)
        self.norm_y = LayerNorm2d(channels)
        self.message_y_to_x = _MessageBranch(channels, hidden_channels)
        self.message_x_to_y = _MessageBranch(channels, hidden_channels)
        self.gate_x = _GateBranch(channels)
        self.gate_y = _GateBranch(channels)
        self.alpha = nn.Parameter(torch.full((1,), float(init_scale)))
        self.beta = nn.Parameter(torch.full((1,), float(init_scale)))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_pair(x, y, self.channels)
        x_norm = self.norm_x(x)
        y_norm = self.norm_y(y)
        difference = x_norm - y_norm

        y_to_x_message = self.message_y_to_x(torch.cat([y_norm, difference], dim=1))
        x_to_y_message = self.message_x_to_y(torch.cat([x_norm, -difference], dim=1))
        x_gate = self.gate_x(torch.cat([x_norm, difference], dim=1))
        y_gate = self.gate_y(torch.cat([y_norm, -difference], dim=1))

        x_out = x + self.alpha * x_gate * y_to_x_message
        y_out = y + self.beta * y_gate * x_to_y_message
        return x_out, y_out


__all__ = ["DGABlockV2", "LayerNorm2d"]
