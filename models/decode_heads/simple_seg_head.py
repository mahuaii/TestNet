from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSegHead(nn.Module):
    """A tiny segmentation head used only to validate the model pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        logits = self.proj(x)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
