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


class StructurePriorModulatedFusionBlock11(nn.Module):
    """Single-scale SPMF11 block with independent modality gates."""

    def __init__(
        self,
        channels: int = 256,
        structure_channels: int | None = None,
        hidden_dim: int = 64,
        *,
        gate_init_std: float = 1e-3,
        norm_layer: type[nn.Module] = LayerNorm2d,
    ) -> None:
        super().__init__()
        self.channels = validate_positive_int("channels", channels)
        self.structure_channels = validate_positive_int(
            "structure_channels",
            self.channels if structure_channels is None else structure_channels,
        )
        self.hidden_dim = validate_positive_int("hidden_dim", hidden_dim)
        self.gate_init_std = float(gate_init_std)
        if self.gate_init_std <= 0:
            raise ValueError(f"Expected gate_init_std to be positive, got {self.gate_init_std}.")

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
        self.rgb_gate_generator = self._make_gate_generator(norm_layer)
        self.dsm_gate_generator = self._make_gate_generator(norm_layer)

    def _make_gate_generator(self, norm_layer: type[nn.Module]) -> nn.Sequential:
        gate_output = nn.Conv2d(self.hidden_dim, self.channels, kernel_size=1)
        nn.init.normal_(gate_output.weight, mean=0.0, std=self.gate_init_std)
        nn.init.zeros_(gate_output.bias)
        return nn.Sequential(
            ConvNormAct(
                self.hidden_dim * 3,
                self.hidden_dim,
                kernel_size=3,
                norm_layer=norm_layer,
            ),
            gate_output,
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
        rgb_gate = torch.sigmoid(self.rgb_gate_generator(gate_input))
        dsm_gate = torch.sigmoid(self.dsm_gate_generator(gate_input))
        return rgb_gate * rgb + dsm_gate * dsm


class MultiScaleStructurePriorModulatedFusion11(nn.Module):
    """Apply independent SPMF11 blocks to four feature scales."""

    def __init__(
        self,
        channels: int | Sequence[int] = 256,
        structure_channels: int | Sequence[int] | None = None,
        hidden_dim: int | Sequence[int] = 64,
        *,
        gate_init_std: float = 1e-3,
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
            StructurePriorModulatedFusionBlock11(
                channels=channel,
                structure_channels=structure_channel,
                hidden_dim=hidden,
                gate_init_std=gate_init_std,
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


SPMFBlock11 = StructurePriorModulatedFusionBlock11
MultiScaleSPMF11 = MultiScaleStructurePriorModulatedFusion11


__all__ = [
    "SPMFBlock11",
    "MultiScaleSPMF11",
    "StructurePriorModulatedFusionBlock11",
    "MultiScaleStructurePriorModulatedFusion11",
]
