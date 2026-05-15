from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dga20 import LayerNorm2d


def _as_channel_tuple(input_channels: int | Sequence[int]) -> tuple[int, int, int, int, int]:
    if isinstance(input_channels, int):
        channels = (input_channels,) * 5
    else:
        channels = tuple(int(channel) for channel in input_channels)
    if len(channels) != 5:
        raise ValueError(f"Expected 5 input channel values, got {len(channels)}.")
    if any(channel <= 0 for channel in channels):
        raise ValueError(f"Expected all input channels to be positive, got {channels}.")
    return channels


def _validate_feature(name: str, value: torch.Tensor, channels: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected {name} to be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim != 4:
        raise ValueError(f"Expected {name} to be 4D with shape [B, C, H, W], got {tuple(value.shape)}.")
    if value.shape[1] != channels:
        raise ValueError(f"Expected {name} channel count {channels}, got {value.shape[1]}.")


def _validate_feature_sequence(name: str, value: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    if not isinstance(value, Sequence):
        raise TypeError(f"Expected {name} to be a sequence of 5 tensors, got {type(value).__name__}.")
    features = tuple(value)
    if len(features) != 5:
        raise ValueError(f"Expected {name} to contain 5 tensors, got {len(features)}.")
    return features


def _resize(x: torch.Tensor, size: tuple[int, int], *, align_corners: bool) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)


class _ConvNormGELU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: type[nn.Module] = LayerNorm2d,
    ) -> None:
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_channels),
            nn.GELU(),
        )


class _LevelFuse(nn.Module):
    def __init__(self, channels: int, norm_layer: type[nn.Module]) -> None:
        super().__init__()
        self.fuse = _ConvNormGELU(channels * 2, channels, kernel_size=1, norm_layer=norm_layer)

    def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([rgb, aux], dim=1))


class _ResizeConv(nn.Module):
    def __init__(self, channels: int, norm_layer: type[nn.Module]) -> None:
        super().__init__()
        self.proj = _ConvNormGELU(channels, channels, kernel_size=1, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor, size: tuple[int, int], *, align_corners: bool) -> torch.Tensor:
        x = _resize(x, size, align_corners=align_corners)
        return self.proj(x)


class _ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        if reduction <= 0:
            raise ValueError(f"Expected reduction to be positive, got {reduction}.")
        hidden_channels = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels * 2, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        gate = torch.cat([lower, upper], dim=1)
        gate = self.pool(gate)
        gate = self.fc1(gate)
        gate = self.act(gate)
        gate = self.fc2(gate)
        return self.gate(gate)


class _GateFuse(nn.Module):
    def __init__(self, channels: int, gate_reduction: int, norm_layer: type[nn.Module]) -> None:
        super().__init__()
        self.gate = _ChannelGate(channels, gate_reduction)
        self.message = _ConvNormGELU(channels, channels, kernel_size=3, norm_layer=norm_layer)

    def forward(self, lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate(lower, upper)
        message = self.message(upper)
        return gate, message


class _ScaleExpansionBranch(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        scale_factor: int | None = None,
        stride: int = 1,
        norm_layer: type[nn.Module],
    ) -> None:
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

    def forward(self, x: torch.Tensor, *, align_corners: bool) -> torch.Tensor:
        if self.scale_factor is not None:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=align_corners)
        return self.proj(x)


class _HRCBlock(nn.Sequential):
    def __init__(self, channels: int, norm_layer: type[nn.Module]) -> None:
        super().__init__(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            norm_layer(channels),
            nn.GELU(),
        )


class DepthGuidedScaleFusion10(nn.Module):
    """
    Depth-Guided Scale Fusion Module 10 (DGSF10).

    The module accepts paired RGB/Aux depth-guided features from four
    intermediate levels plus the final encoder level and returns four
    decoder-ready multi-scale features.
    """

    def __init__(
        self,
        input_channels: int | Sequence[int] = 256,
        hidden_channels: int = 256,
        *,
        gate_reduction: int = 4,
        init_residual_scale: float = 1e-3,
        norm_layer: type[nn.Module] = LayerNorm2d,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError(f"Expected hidden_channels to be positive, got {hidden_channels}.")
        self.input_channels = _as_channel_tuple(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.align_corners = bool(align_corners)

        c = self.hidden_channels
        self.level_fuse1 = _LevelFuse(self.input_channels[0], norm_layer)
        self.level_fuse2 = _LevelFuse(self.input_channels[1], norm_layer)
        self.level_fuse3 = _LevelFuse(self.input_channels[2], norm_layer)
        self.level_fuse4 = _LevelFuse(self.input_channels[3], norm_layer)
        self.level_fuse_top = _LevelFuse(self.input_channels[4], norm_layer)

        self.proj1 = _ConvNormGELU(self.input_channels[0], c, kernel_size=1, norm_layer=norm_layer)
        self.proj2 = _ConvNormGELU(self.input_channels[1], c, kernel_size=1, norm_layer=norm_layer)
        self.proj3 = _ConvNormGELU(self.input_channels[2], c, kernel_size=1, norm_layer=norm_layer)
        self.proj4 = _ConvNormGELU(self.input_channels[3], c, kernel_size=1, norm_layer=norm_layer)
        self.proj_top = _ConvNormGELU(self.input_channels[4], c, kernel_size=1, norm_layer=norm_layer)

        self.gate_fuse1 = _GateFuse(c, gate_reduction, norm_layer)
        self.gate_fuse2 = _GateFuse(c, gate_reduction, norm_layer)
        self.gate_fuse3 = _GateFuse(c, gate_reduction, norm_layer)
        self.gate_fuse4 = _GateFuse(c, gate_reduction, norm_layer)
        self.gamma = nn.Parameter(torch.full((4,), float(init_residual_scale)))

        self.aggregate1 = _ResizeConv(c, norm_layer)
        self.aggregate2 = _ResizeConv(c, norm_layer)
        self.aggregate3 = _ResizeConv(c, norm_layer)
        self.aggregate4 = _ResizeConv(c, norm_layer)
        self.aggregate_top = _ResizeConv(c, norm_layer)
        self.depth_logits = nn.Parameter(torch.zeros(5))

        self.scale_branch1 = _ScaleExpansionBranch(c, scale_factor=4, norm_layer=norm_layer)
        self.scale_branch2 = _ScaleExpansionBranch(c, scale_factor=2, norm_layer=norm_layer)
        self.scale_branch3 = _ScaleExpansionBranch(c, norm_layer=norm_layer)
        self.scale_branch4 = _ScaleExpansionBranch(c, stride=2, norm_layer=norm_layer)

        self.fuse1 = _HRCBlock(c, norm_layer)
        self.fuse2 = _HRCBlock(c, norm_layer)
        self.fuse3 = _HRCBlock(c, norm_layer)
        self.fuse4 = _HRCBlock(c, norm_layer)
        self.delta = nn.Parameter(torch.full((4,), float(init_residual_scale)))

    def forward(
        self,
        rgb_feats: Sequence[torch.Tensor],
        aux_feats: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_feats, aux_feats = self._validate_inputs(rgb_feats, aux_feats)

        f1 = self.level_fuse1(rgb_feats[0], aux_feats[0])
        f2 = self.level_fuse2(rgb_feats[1], aux_feats[1])
        f3 = self.level_fuse3(rgb_feats[2], aux_feats[2])
        f4 = self.level_fuse4(rgb_feats[3], aux_feats[3])
        ftop = self.level_fuse_top(rgb_feats[4], aux_feats[4])

        p1 = self.proj1(f1)
        p2 = self.proj2(f2)
        p3 = self.proj3(f3)
        p4 = self.proj4(f4)
        top = self.proj_top(ftop)

        h4, g4, td4 = self._top_down_step(p4, top, self.gate_fuse4, self.gamma[3])
        h3, g3, td3 = self._top_down_step(p3, h4, self.gate_fuse3, self.gamma[2])
        h2, g2, td2 = self._top_down_step(p2, h3, self.gate_fuse2, self.gamma[1])
        h1, g1, td1 = self._top_down_step(p1, h2, self.gate_fuse1, self.gamma[0])

        base_size = tuple(h1.shape[-2:])
        a1 = self.aggregate1(h1, base_size, align_corners=self.align_corners)
        a2 = self.aggregate2(h2, base_size, align_corners=self.align_corners)
        a3 = self.aggregate3(h3, base_size, align_corners=self.align_corners)
        a4 = self.aggregate4(h4, base_size, align_corners=self.align_corners)
        at = self.aggregate_top(top, base_size, align_corners=self.align_corners)
        weights = torch.softmax(self.depth_logits, dim=0)
        shared = weights[0] * a1 + weights[1] * a2 + weights[2] * a3 + weights[3] * a4 + weights[4] * at

        s1 = self.scale_branch1(shared, align_corners=self.align_corners)
        s2 = self.scale_branch2(shared, align_corners=self.align_corners)
        s3 = self.scale_branch3(shared, align_corners=self.align_corners)
        s4 = self.scale_branch4(shared, align_corners=self.align_corners)

        o1, fd1 = self._hrc(s1, h1, self.fuse1, self.delta[0])
        o2, fd2 = self._hrc(s2, h2, self.fuse2, self.delta[1])
        o3, fd3 = self._hrc(s3, h3, self.fuse3, self.delta[2])
        o4, fd4 = self._hrc(s4, h4, self.fuse4, self.delta[3])

        self._record_debug_stats(
            level_fused_features=(f1, f2, f3, f4, ftop),
            rgb_features=rgb_feats,
            aux_features=aux_feats,
            gates=(g1, g2, g3, g4),
            top_down_injections=(td1, td2, td3, td4),
            projected_features=(p1, p2, p3, p4),
            fuse_injections=(fd1, fd2, fd3, fd4),
            hierarchy_features=(h1, h2, h3, h4),
            depth_weights=weights,
        )
        return o1, o2, o3, o4

    def _validate_inputs(
        self,
        rgb_feats: Sequence[torch.Tensor],
        aux_feats: Sequence[torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        rgb_feats = _validate_feature_sequence("rgb_feats", rgb_feats)
        aux_feats = _validate_feature_sequence("aux_feats", aux_feats)
        level_names = ("1", "2", "3", "4", "top")
        batch_size = (
            rgb_feats[0].shape[0] if isinstance(rgb_feats[0], torch.Tensor) and rgb_feats[0].ndim > 0 else None
        )
        spatial_size = rgb_feats[0].shape[-2:] if isinstance(rgb_feats[0], torch.Tensor) and rgb_feats[0].ndim == 4 else None
        for level_name, rgb, aux, channels in zip(level_names, rgb_feats, aux_feats, self.input_channels):
            rgb_name = f"rgb_feats[{level_name}]"
            aux_name = f"aux_feats[{level_name}]"
            _validate_feature(rgb_name, rgb, channels)
            _validate_feature(aux_name, aux, channels)
            if rgb.shape != aux.shape:
                raise ValueError(
                    f"Expected {rgb_name} and {aux_name} to share shape, got "
                    f"{tuple(rgb.shape)} and {tuple(aux.shape)}."
                )
            if batch_size is not None and rgb.shape[0] != batch_size:
                raise ValueError(
                    f"Expected all DGSF10 inputs to share batch size {batch_size}, "
                    f"but {rgb_name} has batch size {rgb.shape[0]}."
                )
            if spatial_size is not None and tuple(rgb.shape[-2:]) != tuple(spatial_size):
                raise ValueError(
                    f"Expected all DGSF10 inputs to share spatial size {tuple(spatial_size)}, "
                    f"but {rgb_name} has spatial size {tuple(rgb.shape[-2:])}."
                )
        return rgb_feats, aux_feats

    def _top_down_step(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        fuse: _GateFuse,
        gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        upper = _resize(upper, tuple(lower.shape[-2:]), align_corners=self.align_corners)
        gate, message = fuse(lower, upper)
        injection = gamma * gate * message
        return lower + injection, gate, injection

    def _hrc(
        self,
        scale: torch.Tensor,
        hierarchy: torch.Tensor,
        fuse: _HRCBlock,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hierarchy = _resize(hierarchy, tuple(scale.shape[-2:]), align_corners=self.align_corners)
        message = fuse(torch.cat([scale, hierarchy], dim=1))
        injection = delta * message
        return scale + injection, injection

    def _record_debug_stats(
        self,
        *,
        level_fused_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        rgb_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        aux_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        gates: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        top_down_injections: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        projected_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        fuse_injections: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        hierarchy_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        depth_weights: torch.Tensor,
    ) -> None:
        stats = getattr(self, "intermediate_stats", None)
        if stats is None:
            return
        prefix = str(getattr(self, "intermediate_stats_prefix", "dgsf10")).strip("/")
        for name, fused, rgb, aux in zip(
            ("f1", "f2", "f3", "f4", "ftop"),
            level_fused_features,
            rgb_features,
            aux_features,
        ):
            stats.record_mean_std(f"{prefix}/level_fuse/{name}", fused)
            stats.record_norm_ratio(f"{prefix}/level_fuse/{name}_over_rgb", fused, rgb)
            stats.record_norm_ratio(f"{prefix}/level_fuse/{name}_over_aux", fused, aux)
        for index, gate in enumerate(gates, start=1):
            stats.record_mean_std(f"{prefix}/gate/g{index}", gate)
        for index, gamma in enumerate(self.gamma, start=1):
            stats.record_scalar(f"{prefix}/residual/gamma{index}", gamma)
        for index, delta in enumerate(self.delta, start=1):
            stats.record_scalar(f"{prefix}/residual/delta{index}", delta)
        for name, weight in zip(("w1", "w2", "w3", "w4", "wt"), depth_weights):
            stats.record_scalar(f"{prefix}/depth_weight/{name}", weight)
        for index, (injection, projected) in enumerate(zip(top_down_injections, projected_features), start=1):
            stats.record_norm_ratio(f"{prefix}/feature_ratio/topdown{index}_over_p{index}", injection, projected)
        for index, (injection, hierarchy) in enumerate(zip(fuse_injections, hierarchy_features), start=1):
            stats.record_norm_ratio(f"{prefix}/feature_ratio/fuse{index}_over_h{index}", injection, hierarchy)


DGSF10 = DepthGuidedScaleFusion10


__all__ = ["DGSF10", "DepthGuidedScaleFusion10"]
