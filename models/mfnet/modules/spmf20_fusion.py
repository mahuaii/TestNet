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


class SPMFFusionBlock20(nn.Module):
    """Single-scale SPMF20 block with structure-conditioned evidence routing."""

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

        self.rgb_affine = nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=1)
        self.dsm_affine = nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=1)
        self._zero_initialize(self.rgb_affine)
        self._zero_initialize(self.dsm_affine)

        self.rgb_evidence_head = self._make_evidence_head(norm_layer)
        self.dsm_evidence_head = self._make_evidence_head(norm_layer)

    def _make_evidence_head(self, norm_layer: type[nn.Module]) -> nn.Sequential:
        output = nn.Conv2d(self.hidden_dim, self.channels, kernel_size=1)
        self._zero_initialize(output)
        return nn.Sequential(
            ConvNormAct(
                self.hidden_dim * 2,
                self.hidden_dim,
                kernel_size=3,
                norm_layer=norm_layer,
            ),
            output,
        )

    @staticmethod
    def _zero_initialize(module: nn.Conv2d) -> None:
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

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

        rgb_scale, rgb_shift = self.rgb_affine(structure_sem).chunk(2, dim=1)
        dsm_scale, dsm_shift = self.dsm_affine(structure_sem).chunk(2, dim=1)
        rgb_modulated = rgb_sem * (1.0 + rgb_scale) + rgb_shift
        dsm_modulated = dsm_sem * (1.0 + dsm_scale) + dsm_shift

        rgb_logits = self.rgb_evidence_head(torch.cat([rgb_modulated, structure_sem], dim=1))
        dsm_logits = self.dsm_evidence_head(torch.cat([dsm_modulated, structure_sem], dim=1))
        routing_weights = torch.softmax(torch.stack([rgb_logits, dsm_logits], dim=1), dim=1)
        rgb_weight = routing_weights[:, 0]
        dsm_weight = routing_weights[:, 1]
        self._record_gate_stats(rgb_weight=rgb_weight, dsm_weight=dsm_weight)
        return rgb_weight * rgb + dsm_weight * dsm

    def _record_gate_stats(self, *, rgb_weight: torch.Tensor, dsm_weight: torch.Tensor) -> None:
        stats = getattr(self, "intermediate_stats", None)
        if stats is None:
            return
        prefix = str(getattr(self, "intermediate_stats_prefix", "spmf20")).strip("/")
        for name, value in (
            (f"{prefix}/gate/rgb_weight", rgb_weight),
            (f"{prefix}/gate/dsm_weight", dsm_weight),
        ):
            tensor = value.detach()
            stats.record_scalar(f"{name}_mean", tensor.mean())
            stats.record_scalar(f"{name}_std", tensor.std(unbiased=False))
            stats.record_scalar(f"{name}_min", tensor.amin())
            stats.record_scalar(f"{name}_max", tensor.amax())


class MultiScaleSPMFFusion20(nn.Module):
    """Apply independent SPMF20 blocks to four feature scales."""

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
            SPMFFusionBlock20(
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
        outputs = []
        stats = getattr(self, "intermediate_stats", None)
        prefix = str(getattr(self, "intermediate_stats_prefix", "spmf20")).strip("/")
        for index, (block, rgb, dsm, structure) in enumerate(
            zip(
                self.blocks,
                rgb_feats,
                dsm_feats,
                structure_feats,
            ),
            start=1,
        ):
            if stats is not None:
                block.intermediate_stats = stats
                block.intermediate_stats_prefix = f"{prefix}/scale{index}"
            fused = block(rgb, dsm, structure)
            if stats is not None:
                stats.record_norm(f"{prefix}/feature_norm/R{index}", rgb)
                stats.record_norm(f"{prefix}/feature_norm/G{index}", dsm)
                stats.record_norm(f"{prefix}/feature_norm/D{index}", fused)
            outputs.append(fused)
        return tuple(outputs)  # type: ignore[return-value]


__all__ = [
    "SPMFFusionBlock20",
    "MultiScaleSPMFFusion20",
]
