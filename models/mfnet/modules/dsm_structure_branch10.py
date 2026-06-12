from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._spmf_common import (
    ConvNormAct,
    as_four_tuple,
    validate_feature,
    validate_feature_sequence,
    validate_positive_int,
)
from .dga20 import LayerNorm2d


def _validate_dsm(dsm: torch.Tensor) -> torch.Tensor:
    if not isinstance(dsm, torch.Tensor):
        raise TypeError(f"Expected dsm to be a torch.Tensor, got {type(dsm).__name__}.")
    if dsm.ndim == 3:
        dsm = dsm.unsqueeze(1)
    if dsm.ndim != 4:
        raise ValueError(f"Expected dsm to be 3D [B, H, W] or 4D [B, 1, H, W], got {tuple(dsm.shape)}.")
    if dsm.shape[1] != 1:
        raise ValueError(f"Expected dsm channel count 1, got {dsm.shape[1]}.")
    return dsm


def _resize(x: torch.Tensor, size: tuple[int, int], *, align_corners: bool) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)


class _StructureStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_layer: type[nn.Module]) -> None:
        super().__init__()
        self.down = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=2, norm_layer=norm_layer)
        self.refine = ConvNormAct(out_channels, out_channels, kernel_size=3, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.down(x))


class _TapAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_layer: type[nn.Module]) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size=1, norm_layer=norm_layer),
            ConvNormAct(out_channels, out_channels, kernel_size=3, norm_layer=norm_layer),
        )

    def forward(self, x: torch.Tensor, size: tuple[int, int], *, align_corners: bool) -> torch.Tensor:
        x = _resize(x, size, align_corners=align_corners)
        return self.proj(x)


class DSMStructureBranch10(nn.Module):
    """Generate multi-scale structure features from DSM and DSM SAM taps."""

    def __init__(
        self,
        tap_channels: int | Sequence[int] = 256,
        structure_channels: Sequence[int] = (64, 96, 128, 160),
        output_channels: int = 256,
        *,
        similarity_kernel_size: int = 7,
        similarity_sigma: float = 0.15,
        eps: float = 1e-6,
        norm_layer: type[nn.Module] = LayerNorm2d,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.tap_channels = as_four_tuple("tap_channels", tap_channels)
        self.structure_channels = as_four_tuple("structure_channels", structure_channels)
        self.output_channels = validate_positive_int("output_channels", output_channels)
        self.similarity_kernel_size = int(similarity_kernel_size)
        if self.similarity_kernel_size <= 0 or self.similarity_kernel_size % 2 == 0:
            raise ValueError(
                "Expected similarity_kernel_size to be a positive odd integer, "
                f"got {self.similarity_kernel_size}."
            )
        self.similarity_sigma = float(similarity_sigma)
        if self.similarity_sigma <= 0:
            raise ValueError(f"Expected similarity_sigma to be positive, got {self.similarity_sigma}.")
        self.eps = float(eps)
        self.align_corners = bool(align_corners)

        c1, c2, c3, c4 = self.structure_channels
        self.stem = nn.Sequential(
            ConvNormAct(2, c1, kernel_size=3, stride=2, norm_layer=norm_layer),
            ConvNormAct(c1, c1, kernel_size=3, norm_layer=norm_layer),
        )
        self.stage1 = _StructureStage(c1, c1, norm_layer)
        self.stage2 = _StructureStage(c1, c2, norm_layer)
        self.stage3 = _StructureStage(c2, c3, norm_layer)
        self.stage4 = _StructureStage(c3, c4, norm_layer)

        self.tap_adapters = nn.ModuleList(
            _TapAdapter(tap_channel, structure_channel, norm_layer)
            for tap_channel, structure_channel in zip(self.tap_channels, self.structure_channels)
        )
        self.confidence_generators = nn.ModuleList(
            nn.Sequential(
                ConvNormAct(structure_channel, structure_channel, kernel_size=3, norm_layer=norm_layer),
                nn.Conv2d(structure_channel, self.output_channels, kernel_size=1),
                nn.Sigmoid(),
            )
            for structure_channel in self.structure_channels
        )
        self.output_projections = nn.ModuleList(
            ConvNormAct(structure_channel, self.output_channels, kernel_size=1, norm_layer=norm_layer)
            for structure_channel in self.structure_channels
        )

    def forward(
        self,
        dsm: torch.Tensor,
        dsm_taps: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dsm = _validate_dsm(dsm)
        dsm_taps = validate_feature_sequence("dsm_taps", dsm_taps)
        for index, (tap, channels) in enumerate(zip(dsm_taps, self.tap_channels)):
            validate_feature(f"dsm_taps[{index}]", tap, channels)
            if tap.shape[0] != dsm.shape[0]:
                raise ValueError(
                    f"Expected dsm_taps[{index}] batch size {dsm.shape[0]}, got {tap.shape[0]}."
                )

        h, w = dsm.shape[-2:]
        target_sizes = tuple((max(1, h // scale), max(1, w // scale)) for scale in (4, 8, 16, 32))
        structure_input = self._make_structure_input(dsm)

        x = self.stem(structure_input)
        g1 = _resize(self.stage1(x), target_sizes[0], align_corners=self.align_corners)
        g2 = _resize(self.stage2(g1), target_sizes[1], align_corners=self.align_corners)
        g3 = _resize(self.stage3(g2), target_sizes[2], align_corners=self.align_corners)
        g4 = _resize(self.stage4(g3), target_sizes[3], align_corners=self.align_corners)
        geometry_features = (g1, g2, g3, g4)

        outputs = []
        for geometry, tap, tap_adapter, confidence_generator, projection in zip(
            geometry_features,
            dsm_taps,
            self.tap_adapters,
            self.confidence_generators,
            self.output_projections,
        ):
            tap_for_structure = tap.detach()
            adapted_tap = tap_adapter(
                tap_for_structure,
                tuple(geometry.shape[-2:]),
                align_corners=self.align_corners,
            )
            confidence = confidence_generator(adapted_tap)
            outputs.append(projection(geometry) * (1.0 + confidence))
        return tuple(outputs)  # type: ignore[return-value]

    def _make_structure_input(self, dsm: torch.Tensor) -> torch.Tensor:
        dsm_norm = self._normalize_dsm(dsm)
        similarity = self._local_similarity(dsm_norm)
        return torch.cat([dsm_norm, similarity], dim=1)

    def _normalize_dsm(self, dsm: torch.Tensor) -> torch.Tensor:
        dsm_min = dsm.amin(dim=(-2, -1), keepdim=True)
        dsm_max = dsm.amax(dim=(-2, -1), keepdim=True)
        return (dsm - dsm_min) / (dsm_max - dsm_min).clamp_min(self.eps)

    def _local_similarity(self, dsm_norm: torch.Tensor) -> torch.Tensor:
        kernel_size = self.similarity_kernel_size
        padding = kernel_size // 2
        local_mean = F.avg_pool2d(
            dsm_norm,
            kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        local_square_mean = F.avg_pool2d(
            dsm_norm.square(),
            kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        local_variance = (local_square_mean - local_mean.square()).clamp_min(0.0)
        return torch.exp(-local_variance / (2.0 * self.similarity_sigma * self.similarity_sigma))


__all__ = ["DSMStructureBranch10"]
