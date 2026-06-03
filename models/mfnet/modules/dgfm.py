import torch
import torch.nn as nn
import torch.nn.functional as F

from .dga20 import LayerNorm2d


def _validate_feature(name, value):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected {name} to be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim != 4:
        raise ValueError(f"Expected {name} to be 4D, got shape {tuple(value.shape)}.")


class DGFMScaleAdapter(nn.Module):
    """Expand one DGFM feature map into decoder-ready four-scale features."""

    def __init__(self, channels, norm_layer=LayerNorm2d, align_corners=False):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"Expected channels to be positive, got {channels}.")
        self.channels = int(channels)
        self.align_corners = bool(align_corners)
        self.res1 = _ResizeConvBranch(channels, scale_factor=4, norm_layer=norm_layer)
        self.res2 = _ResizeConvBranch(channels, scale_factor=2, norm_layer=norm_layer)
        self.res3 = _ResizeConvBranch(channels, norm_layer=norm_layer)
        self.res4 = _ResizeConvBranch(channels, stride=2, norm_layer=norm_layer)

    def forward(self, x):
        _validate_feature("x", x)
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected x channel count {self.channels}, got {x.shape[1]}.")
        return (
            self.res1(x, align_corners=self.align_corners),
            self.res2(x, align_corners=self.align_corners),
            self.res3(x, align_corners=self.align_corners),
            self.res4(x, align_corners=self.align_corners),
        )


class _ResizeConvBranch(nn.Module):
    def __init__(self, channels, *, scale_factor=None, stride=1, norm_layer=LayerNorm2d):
        super().__init__()
        if scale_factor is not None and scale_factor <= 0:
            raise ValueError(f"Expected scale_factor to be positive, got {scale_factor}.")
        if stride <= 0:
            raise ValueError(f"Expected stride to be positive, got {stride}.")
        self.scale_factor = scale_factor
        self.proj = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            norm_layer(channels),
            nn.GELU(),
        )

    def forward(self, x, *, align_corners):
        if self.scale_factor is not None:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=align_corners)
        return self.proj(x)


class DGFM(nn.Module):
    def __init__(self, dims, ratio=0.25, out_channels=256):
        super().__init__()
        hidden_dim = max(32, int(dims * min(ratio, 0.125)))

        self.input_norm = nn.LayerNorm(dims)
        self.rgb_reduce = nn.Conv2d(dims, hidden_dim, kernel_size=1, bias=False)
        self.dsm_reduce = nn.Conv2d(dims, hidden_dim, kernel_size=1, bias=False)
        self.gate_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dims, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        nn.init.zeros_(self.gate_net[-2].weight)
        nn.init.zeros_(self.gate_net[-2].bias)
        self.output_norm = nn.LayerNorm(dims)
        self.output_proj = nn.Conv2d(dims, out_channels, kernel_size=1, bias=False)
        self.scale_adapter = DGFMScaleAdapter(out_channels)

    def forward(self, rgb, dsm):
        if rgb.shape != dsm.shape:
            raise ValueError(
                f"Expected rgb and dsm to have the same shape, got {tuple(rgb.shape)} and {tuple(dsm.shape)}."
            )
        rgb = self.input_norm(rgb).permute(0, 3, 1, 2).contiguous()
        dsm = self.input_norm(dsm).permute(0, 3, 1, 2).contiguous()

        rgb_reduce = self.rgb_reduce(rgb)
        dsm_reduce = self.dsm_reduce(dsm)
        diff_reduce = torch.abs(rgb_reduce - dsm_reduce)

        gate = self.gate_net(
            torch.cat([rgb_reduce, dsm_reduce, diff_reduce], dim=1)
        )
        fused = gate * rgb + (1.0 - gate) * dsm
        fused = self.output_norm(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        fused = self.output_proj(fused)
        return self.scale_adapter(fused)


__all__ = ["DGFM", "DGFMScaleAdapter"]
