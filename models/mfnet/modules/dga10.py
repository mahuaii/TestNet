from __future__ import annotations

import torch
import torch.nn as nn


def _hidden_channels(channels: int, reduction: int) -> int:
    if channels <= 0:
        raise ValueError(f"Expected channels to be positive, got {channels}.")
    if reduction <= 0:
        raise ValueError(f"Expected reduction to be positive, got {reduction}.")
    return max(1, channels // reduction)


def _validate_pair(rgb: torch.Tensor, aux: torch.Tensor, channels: int) -> None:
    if rgb.ndim != 4 or aux.ndim != 4:
        raise ValueError("Expected rgb and aux to be 4D tensors with shape [B, C, H, W].")
    if rgb.shape != aux.shape:
        raise ValueError(f"Expected rgb and aux to have the same shape, got {tuple(rgb.shape)} and {tuple(aux.shape)}.")
    if rgb.shape[1] != channels:
        raise ValueError(f"Expected input channel count {channels}, got {rgb.shape[1]}.")


class _MessageBranch(nn.Sequential):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )


class _GateBranch(nn.Sequential):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )


class DGABlock10(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, init_scale: float = 1e-3) -> None:
        super().__init__()
        hidden_channels = _hidden_channels(channels, reduction)
        self.channels = int(channels)
        self.reduction = int(reduction)
        self.init_scale = float(init_scale)

        self.norm_rgb = nn.BatchNorm2d(channels)
        self.norm_aux = nn.BatchNorm2d(channels)
        self.message_aux_to_rgb = _MessageBranch(channels, hidden_channels)
        self.message_rgb_to_aux = _MessageBranch(channels, hidden_channels)
        self.gate_rgb = _GateBranch(channels, hidden_channels)
        self.gate_aux = _GateBranch(channels, hidden_channels)
        self.alpha = nn.Parameter(torch.tensor(float(init_scale)))
        self.beta = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_pair(rgb, aux, self.channels)
        rgb_norm = self.norm_rgb(rgb)
        aux_norm = self.norm_aux(aux)
        difference = rgb_norm - aux_norm

        aux_to_rgb_message = self.message_aux_to_rgb(torch.cat([aux_norm, difference], dim=1))
        rgb_to_aux_message = self.message_rgb_to_aux(torch.cat([rgb_norm, -difference], dim=1))
        rgb_gate = self.gate_rgb(torch.cat([rgb_norm, difference], dim=1))
        aux_gate = self.gate_aux(torch.cat([aux_norm, -difference], dim=1))

        rgb_out = rgb + self.alpha * rgb_gate * aux_to_rgb_message
        aux_out = aux + self.beta * aux_gate * rgb_to_aux_message
        return rgb_out, aux_out


__all__ = ["DGABlock10"]
