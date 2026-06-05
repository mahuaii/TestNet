from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dga20 import LayerNorm2d
from .scale_adapter import DGFMScaleAdapter


def _validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"Expected {name} to be positive, got {value}.")
    return value


def _validate_group_norm_channels(channels: int, groups: int) -> None:
    if channels % groups != 0:
        raise ValueError(f"Expected channels {channels} to be divisible by groups {groups}.")


def _validate_feature(name: str, value: torch.Tensor, channels: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected {name} to be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim != 4:
        raise ValueError(f"Expected {name} to be 4D with shape [B, C, H, W], got {tuple(value.shape)}.")
    if value.shape[1] != channels:
        raise ValueError(f"Expected {name} channel count {channels}, got {value.shape[1]}.")


def _validate_dsm_edge(value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected dsm_edge to be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim != 4:
        raise ValueError(f"Expected dsm_edge to be 4D with shape [B, 1, H, W], got {tuple(value.shape)}.")
    if value.shape[1] != 1:
        raise ValueError(f"Expected dsm_edge channel count 1, got {value.shape[1]}.")


def _validate_bhwc_feature(name: str, value: torch.Tensor, channels: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected {name} to be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim != 4:
        raise ValueError(f"Expected {name} to be 4D with shape [B, H, W, C], got {tuple(value.shape)}.")
    if value.shape[-1] != channels:
        raise ValueError(f"Expected {name} channel count {channels}, got {value.shape[-1]}.")


def _validate_feature_sequence(name: str, value: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    if not isinstance(value, Sequence):
        raise TypeError(f"Expected {name} to be a sequence of 4 tensors, got {type(value).__name__}.")
    features = tuple(value)
    if len(features) != 4:
        raise ValueError(f"Expected {name} to contain 4 tensors, got {len(features)}.")
    return features


class _ConvGNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        groups: int,
        conv_groups: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=conv_groups,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )


class SGCFBlock(nn.Module):
    """Single-scale Structure-Guided Compensation Fusion block."""

    def __init__(
        self,
        in_channels: int,
        aux_channels: int | None = None,
        hidden_dim: int = 64,
        groups: int = 8,
        init_residual_scale: float = 1e-3,
        eps: float = 1e-6,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = _validate_positive_int("in_channels", in_channels)
        self.aux_channels = _validate_positive_int(
            "aux_channels",
            self.in_channels if aux_channels is None else aux_channels,
        )
        self.hidden_dim = _validate_positive_int("hidden_dim", hidden_dim)
        self.groups = _validate_positive_int("groups", groups)
        self.eps = float(eps)
        self.align_corners = bool(align_corners)
        _validate_group_norm_channels(self.hidden_dim, self.groups)
        _validate_group_norm_channels(self.hidden_dim * 2, self.groups)

        if self.aux_channels == self.in_channels:
            self.aux_match = nn.Identity()
        else:
            self.aux_match = nn.Conv2d(self.aux_channels, self.in_channels, kernel_size=1, bias=False)

        self.rgb_proj = _ConvGNAct(self.in_channels, self.hidden_dim, kernel_size=1, groups=self.groups)
        self.aux_proj = _ConvGNAct(self.in_channels, self.hidden_dim, kernel_size=1, groups=self.groups)
        self.edge_proj = _ConvGNAct(1, self.hidden_dim, kernel_size=1, groups=self.groups)

        self.aux_struct = _ConvGNAct(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            groups=self.groups,
            conv_groups=self.hidden_dim,
        )
        self.struct_fuse = _ConvGNAct(self.hidden_dim * 2, self.hidden_dim, kernel_size=1, groups=self.groups)

        self.rgb_query = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self.groups, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        gate_hidden = max(1, self.hidden_dim // 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.hidden_dim, gate_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(gate_hidden, self.in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.compensation = nn.Sequential(
            nn.Conv2d(
                self.hidden_dim * 2,
                self.hidden_dim * 2,
                kernel_size=3,
                padding=1,
                groups=self.hidden_dim * 2,
                bias=False,
            ),
            nn.GroupNorm(self.groups, self.hidden_dim * 2),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim * 2, self.in_channels, kernel_size=1),
        )
        self.gamma = nn.Parameter(torch.full((1,), float(init_residual_scale)))

    def forward(self, rgb_feat: torch.Tensor, aux_feat: torch.Tensor, dsm_edge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_feature("rgb_feat", rgb_feat, self.in_channels)
        _validate_feature("aux_feat", aux_feat, self.aux_channels)
        _validate_dsm_edge(dsm_edge)
        if rgb_feat.shape[0] != aux_feat.shape[0]:
            raise ValueError(
                f"Expected rgb_feat and aux_feat to share batch size, got {rgb_feat.shape[0]} and {aux_feat.shape[0]}."
            )
        if tuple(rgb_feat.shape[-2:]) != tuple(aux_feat.shape[-2:]):
            raise ValueError(
                "Expected rgb_feat and aux_feat to share spatial size, got "
                f"{tuple(rgb_feat.shape[-2:])} and {tuple(aux_feat.shape[-2:])}."
            )
        if dsm_edge.shape[0] != rgb_feat.shape[0]:
            raise ValueError(
                f"Expected dsm_edge batch size {rgb_feat.shape[0]}, got {dsm_edge.shape[0]}."
            )

        edge = F.interpolate(
            dsm_edge,
            size=tuple(rgb_feat.shape[-2:]),
            mode="bilinear",
            align_corners=self.align_corners,
        )
        aux_feat = self.aux_match(aux_feat)

        r = self.rgb_proj(rgb_feat)
        a = self.aux_proj(aux_feat)
        edge = self.edge_proj(edge)
        structure = self.struct_fuse(torch.cat([self.aux_struct(a), edge], dim=1))
        query = self.rgb_query(r)
        prior = query * structure

        mask = self.spatial_gate(torch.cat([structure, prior], dim=1))
        channel_gate = self.channel_gate(prior)
        compensation = self.compensation(torch.cat([a, structure], dim=1))
        output = rgb_feat + self.gamma * mask * channel_gate * compensation
        return output, mask


class SobelDSMEdge(nn.Module):
    """Fixed Sobel edge extractor for single-channel DSM inputs."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer(
            "kernel_x",
            torch.tensor(
                [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
                dtype=torch.float32,
            ).unsqueeze(0),
        )
        self.register_buffer(
            "kernel_y",
            torch.tensor(
                [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
                dtype=torch.float32,
            ).unsqueeze(0),
        )

    def forward(self, dsm: torch.Tensor) -> torch.Tensor:
        if not isinstance(dsm, torch.Tensor):
            raise TypeError(f"Expected dsm to be a torch.Tensor, got {type(dsm).__name__}.")
        if dsm.ndim == 3:
            dsm = dsm.unsqueeze(1)
        _validate_dsm_edge(dsm)
        kernel_x = self.kernel_x.to(dtype=dsm.dtype)
        kernel_y = self.kernel_y.to(dtype=dsm.dtype)
        grad_x = F.conv2d(dsm, kernel_x, padding=1)
        grad_y = F.conv2d(dsm, kernel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + self.eps)


class SGCF(nn.Module):
    """DGFM-style SGCF wrapper for one BHWC encoder tap."""

    def __init__(
        self,
        dims: int,
        hidden_dim: int = 64,
        out_channels: int = 256,
        groups: int = 8,
        init_residual_scale: float = 1e-3,
        eps: float = 1e-6,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.dims = _validate_positive_int("dims", dims)
        self.out_channels = _validate_positive_int("out_channels", out_channels)
        self.align_corners = bool(align_corners)

        self.input_norm = nn.LayerNorm(self.dims)
        self.edge = SobelDSMEdge(eps=eps)
        self.fusion = SGCFBlock(
            self.dims,
            hidden_dim=hidden_dim,
            groups=groups,
            init_residual_scale=init_residual_scale,
            eps=eps,
            align_corners=align_corners,
        )
        self.output_norm = nn.LayerNorm(self.dims)
        self.output_proj = nn.Conv2d(self.dims, self.out_channels, kernel_size=1, bias=False)
        self.scale_adapter = DGFMScaleAdapter(self.out_channels)

    def forward(
        self,
        rgb: torch.Tensor,
        aux: torch.Tensor,
        dsm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_bhwc_feature("rgb", rgb, self.dims)
        _validate_bhwc_feature("aux", aux, self.dims)
        if rgb.shape != aux.shape:
            raise ValueError(
                f"Expected rgb and aux to have the same shape, got {tuple(rgb.shape)} and {tuple(aux.shape)}."
            )

        rgb = self.input_norm(rgb).permute(0, 3, 1, 2).contiguous()
        aux = self.input_norm(aux).permute(0, 3, 1, 2).contiguous()
        dsm_edge = self.edge(dsm)
        fused, mask = self.fusion(rgb, aux, dsm_edge)
        self.last_spatial_mask = mask.detach()
        fused = self.output_norm(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        fused = self.output_proj(fused)
        return self.scale_adapter(fused)


class _StageScaleBranch(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        scale_factor: int | None = None,
        stride: int = 1,
        norm_layer: type[nn.Module] = LayerNorm2d,
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


class SGCFScaleAdapter(nn.Module):
    """Adapt four independently fused SGCF tap features to decoder scales."""

    def __init__(
        self,
        channels: int,
        norm_layer: type[nn.Module] = LayerNorm2d,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.channels = _validate_positive_int("channels", channels)
        self.align_corners = bool(align_corners)
        self.res1 = _StageScaleBranch(self.channels, scale_factor=4, norm_layer=norm_layer)
        self.res2 = _StageScaleBranch(self.channels, scale_factor=2, norm_layer=norm_layer)
        self.res3 = _StageScaleBranch(self.channels, norm_layer=norm_layer)
        self.res4 = _StageScaleBranch(self.channels, stride=2, norm_layer=norm_layer)

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = _validate_feature_sequence("features", features)
        for index, feature in enumerate(features):
            _validate_feature(f"features[{index}]", feature, self.channels)
        return (
            self.res1(features[0], align_corners=self.align_corners),
            self.res2(features[1], align_corners=self.align_corners),
            self.res3(features[2], align_corners=self.align_corners),
            self.res4(features[3], align_corners=self.align_corners),
        )


class MultiScaleSGCF(nn.Module):
    """Apply four independent SGCF blocks to four feature scales."""

    def __init__(
        self,
        channels: Sequence[int] = (256, 256, 256, 256),
        aux_channels: Sequence[int] | None = None,
        hidden_dim: int = 64,
        groups: int = 8,
        init_residual_scale: float = 1e-3,
        eps: float = 1e-6,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        channels = tuple(_validate_positive_int("channels", channel) for channel in channels)
        if len(channels) != 4:
            raise ValueError(f"Expected channels to contain 4 values, got {len(channels)}.")
        if aux_channels is None:
            aux_channels = channels
        else:
            aux_channels = tuple(_validate_positive_int("aux_channels", channel) for channel in aux_channels)
        if len(aux_channels) != 4:
            raise ValueError(f"Expected aux_channels to contain 4 values, got {len(aux_channels)}.")

        self.channels = channels
        self.aux_channels = tuple(aux_channels)
        self.blocks = nn.ModuleList(
            [
                SGCFBlock(
                    channel,
                    aux_channel,
                    hidden_dim=hidden_dim,
                    groups=groups,
                    init_residual_scale=init_residual_scale,
                    eps=eps,
                    align_corners=align_corners,
                )
                for channel, aux_channel in zip(self.channels, self.aux_channels)
            ]
        )

    def forward(
        self,
        rgb_feats: Sequence[torch.Tensor],
        aux_feats: Sequence[torch.Tensor],
        dsm_edge: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        rgb_feats = _validate_feature_sequence("rgb_feats", rgb_feats)
        aux_feats = _validate_feature_sequence("aux_feats", aux_feats)
        outputs = []
        masks = []
        for block, rgb_feat, aux_feat in zip(self.blocks, rgb_feats, aux_feats):
            output, mask = block(rgb_feat, aux_feat, dsm_edge)
            outputs.append(output)
            masks.append(mask)
        return tuple(outputs), tuple(masks)


__all__ = ["SGCF", "SGCFBlock", "SGCFScaleAdapter", "MultiScaleSGCF", "SobelDSMEdge"]
