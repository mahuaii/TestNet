from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .dga20 import LayerNorm2d


def validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"Expected {name} to be positive, got {value}.")
    return value


def as_four_tuple(name: str, value: int | Sequence[int]) -> tuple[int, int, int, int]:
    if isinstance(value, int):
        values = (value,) * 4
    else:
        values = tuple(int(item) for item in value)
    if len(values) != 4:
        raise ValueError(f"Expected {name} to contain 4 values, got {len(values)}.")
    for index, item in enumerate(values):
        validate_positive_int(f"{name}[{index}]", item)
    return values


def validate_feature(name: str, value: torch.Tensor, channels: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected {name} to be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim != 4:
        raise ValueError(f"Expected {name} to be 4D with shape [B, C, H, W], got {tuple(value.shape)}.")
    if value.shape[1] != channels:
        raise ValueError(f"Expected {name} channel count {channels}, got {value.shape[1]}.")


def validate_feature_sequence(name: str, value: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    if not isinstance(value, Sequence):
        raise TypeError(f"Expected {name} to be a sequence of 4 tensors, got {type(value).__name__}.")
    features = tuple(value)
    if len(features) != 4:
        raise ValueError(f"Expected {name} to contain 4 tensors, got {len(features)}.")
    return features


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        norm_layer: type[nn.Module] = LayerNorm2d,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_channels),
            nn.GELU(),
        )
