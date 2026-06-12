from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from ._spmf_common import (
    ConvNormAct,
    as_four_tuple,
    validate_feature,
    validate_feature_sequence,
    validate_positive_int,
)
from .dga20 import LayerNorm2d


class StructurePriorModulatedFusionBlock10(nn.Module):
    """Single-scale SPMF10 block."""

    def __init__(
        self,
        channels: int = 256,
        structure_channels: int | None = None,
        hidden_dim: int = 64,
        *,
        norm_layer: type[nn.Module] = LayerNorm2d,
    ) -> None:
        super().__init__()
        self.channels = validate_positive_int("channels", channels)
        self.structure_channels = validate_positive_int(
            "structure_channels",
            self.channels if structure_channels is None else structure_channels,
        )
        self.hidden_dim = validate_positive_int("hidden_dim", hidden_dim)

        self.rgb_projection = ConvNormAct(
            self.channels,
            self.hidden_dim,
            kernel_size=1,
            norm_layer=norm_layer,
        )
        self.dsm_projection = ConvNormAct(
            self.channels,
            self.hidden_dim,
            kernel_size=1,
            norm_layer=norm_layer,
        )
        self.structure_projection = ConvNormAct(
            self.structure_channels,
            self.hidden_dim,
            kernel_size=1,
            norm_layer=norm_layer,
        )

        fusion_gate_output = nn.Conv2d(self.hidden_dim, self.channels, kernel_size=1)
        nn.init.zeros_(fusion_gate_output.weight)
        nn.init.zeros_(fusion_gate_output.bias)
        self.fusion_gate_generator = nn.Sequential(
            ConvNormAct(
                self.hidden_dim * 3,
                self.hidden_dim,
                kernel_size=3,
                norm_layer=norm_layer,
            ),
            fusion_gate_output,
        )

    def forward(self, rgb: torch.Tensor, dsm: torch.Tensor, structure: torch.Tensor) -> torch.Tensor:
        validate_feature("rgb", rgb, self.channels)
        validate_feature("dsm", dsm, self.channels)
        validate_feature("structure", structure, self.structure_channels)
        if rgb.shape != dsm.shape:
            raise ValueError(
                "Expected rgb and dsm to have the same shape, got "
                f"{tuple(rgb.shape)} and {tuple(dsm.shape)}."
            )
        if structure.shape[0] != rgb.shape[0]:
            raise ValueError(f"Expected structure batch size {rgb.shape[0]}, got {structure.shape[0]}.")
        if tuple(structure.shape[-2:]) != tuple(rgb.shape[-2:]):
            raise ValueError(
                "Expected structure spatial size to match rgb/dsm, got "
                f"{tuple(structure.shape[-2:])} and {tuple(rgb.shape[-2:])}."
            )

        rgb_sem = self.rgb_projection(rgb)
        dsm_sem = self.dsm_projection(dsm)
        structure_sem = self.structure_projection(structure)

        gate_input = torch.cat([rgb_sem, dsm_sem, structure_sem], dim=1)
        fusion_gate = torch.sigmoid(self.fusion_gate_generator(gate_input))
        return fusion_gate * rgb + (1.0 - fusion_gate) * dsm


class MultiScaleStructurePriorModulatedFusion10(nn.Module):
    """Apply SPMF10 to four feature scales."""

    def __init__(
        self,
        channels: int | Sequence[int] = 256,
        structure_channels: int | Sequence[int] | None = None,
        hidden_dim: int | Sequence[int] = 64,
        *,
        norm_layer: type[nn.Module] = LayerNorm2d,
    ) -> None:
        super().__init__()
        self.channels = as_four_tuple("channels", channels)
        if structure_channels is None:
            self.structure_channels = self.channels
        else:
            self.structure_channels = as_four_tuple("structure_channels", structure_channels)
        self.hidden_dim = as_four_tuple("hidden_dim", hidden_dim)
        self.blocks = nn.ModuleList(
            StructurePriorModulatedFusionBlock10(
                channels=channel,
                structure_channels=structure_channel,
                hidden_dim=hidden,
                norm_layer=norm_layer,
            )
            for channel, structure_channel, hidden in zip(
                self.channels,
                self.structure_channels,
                self.hidden_dim,
            )
        )

    def forward(
        self,
        rgb_feats: Sequence[torch.Tensor],
        dsm_feats: Sequence[torch.Tensor],
        structure_feats: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_feats = validate_feature_sequence("rgb_feats", rgb_feats)
        dsm_feats = validate_feature_sequence("dsm_feats", dsm_feats)
        structure_feats = validate_feature_sequence("structure_feats", structure_feats)
        outputs = [
            block(rgb, dsm, structure)
            for block, rgb, dsm, structure in zip(
                self.blocks,
                rgb_feats,
                dsm_feats,
                structure_feats,
            )
        ]
        return tuple(outputs)  # type: ignore[return-value]


SPMFBlock10 = StructurePriorModulatedFusionBlock10
MultiScaleSPMF10 = MultiScaleStructurePriorModulatedFusion10


__all__ = [
    "SPMFBlock10",
    "MultiScaleSPMF10",
    "StructurePriorModulatedFusionBlock10",
    "MultiScaleStructurePriorModulatedFusion10",
]
