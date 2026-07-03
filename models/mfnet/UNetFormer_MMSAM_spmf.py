from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .UNetFormer_MMSAM import UNetFormer
from .intermediate_stats_config import attach_requested_intermediate_stats
from .modules import (
    DSMStructureBranch10,
    DSMStructureBranch11,
    DSMStructureBranch12,
    DSMStructureBranch13,
    MultiScaleSPMFFusion10,
    MultiScaleSPMFFusion11,
    MultiScaleSPMFFusion20,
    MultiScaleSPMFFusion21,
    MultiScaleSPMFFusion22,
)


@dataclass(frozen=True)
class SPMFVariantSpec:
    variant_name: str
    structure_branch_cls: type[nn.Module]
    fusion_cls: type[nn.Module]
    indexes_attr: str
    structure_attr: str
    fusion_attr: str
    intermediate_modules: tuple[tuple[str, str, str], ...]


SPMF_VARIANTS: dict[str, SPMFVariantSpec] = {
    "10": SPMFVariantSpec(
        variant_name="10",
        structure_branch_cls=DSMStructureBranch10,
        fusion_cls=MultiScaleSPMFFusion10,
        indexes_attr="spmf10_indexes",
        structure_attr="structure_branch10",
        fusion_attr="spmf_fusion10",
        intermediate_modules=(
            ("spmf_fusion10", "spmf_fusion10", "spmf_fusion10"),
            ("structure10", "structure_branch10", "spmf10/structure"),
        ),
    ),
    "11": SPMFVariantSpec(
        variant_name="11",
        structure_branch_cls=DSMStructureBranch11,
        fusion_cls=MultiScaleSPMFFusion11,
        indexes_attr="spmf11_indexes",
        structure_attr="structure_branch11",
        fusion_attr="spmf_fusion11",
        intermediate_modules=(
            ("spmf_fusion11", "spmf_fusion11", "spmf_fusion11"),
            ("structure11", "structure_branch11", "spmf11/structure"),
        ),
    ),
    "20": SPMFVariantSpec(
        variant_name="20",
        structure_branch_cls=DSMStructureBranch10,
        fusion_cls=MultiScaleSPMFFusion20,
        indexes_attr="spmf20_indexes",
        structure_attr="structure_branch10",
        fusion_attr="spmf_fusion20",
        intermediate_modules=(
            ("spmf_fusion20", "spmf_fusion20", "spmf_fusion20"),
            ("structure10", "structure_branch10", "spmf20/structure"),
        ),
    ),
    "21": SPMFVariantSpec(
        variant_name="21",
        structure_branch_cls=DSMStructureBranch12,
        fusion_cls=MultiScaleSPMFFusion21,
        indexes_attr="spmf21_indexes",
        structure_attr="structure_branch12",
        fusion_attr="spmf_fusion21",
        intermediate_modules=(
            ("spmf_fusion21", "spmf_fusion21", "spmf_fusion21"),
            ("structure12", "structure_branch12", "spmf21/structure"),
            ("structure21", "structure_branch12", "spmf21/structure"),
        ),
    ),
    "22": SPMFVariantSpec(
        variant_name="22",
        structure_branch_cls=DSMStructureBranch13,
        fusion_cls=MultiScaleSPMFFusion22,
        indexes_attr="spmf22_indexes",
        structure_attr="structure_branch13",
        fusion_attr="spmf_fusion22",
        intermediate_modules=(
            ("spmf_fusion22", "spmf_fusion22", "spmf_fusion22"),
            ("structure13", "structure_branch13", "spmf22/structure"),
            ("structure22", "structure_branch13", "spmf22/structure"),
        ),
    ),
}


def _resolve_spmf_indexes(image_encoder: nn.Module, variant_name: str) -> list[int]:
    global_indexes = [
        index for index, block in enumerate(image_encoder.blocks) if getattr(block, "window_size", None) == 0
    ]
    if len(global_indexes) != 4:
        raise ValueError(
            f"Expected exactly 4 encoder feature taps from global attention blocks for SPMF{variant_name}, "
            f"got {len(global_indexes)}: {global_indexes}."
        )
    return [int(index) for index in global_indexes]


def _as_single_channel_dsm(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 3:
        return y.unsqueeze(1)
    if y.ndim == 4 and y.shape[1] == 1:
        return y
    raise ValueError(f"Expected DSM with shape [B, H, W] or [B, 1, H, W], got {tuple(y.shape)}.")


class UNetFormerSPMF(UNetFormer):
    spmf_variant: str

    def __init__(
        self,
        *args: object,
        record_intermediate_stats: bool = False,
        record_intermediate_modules: Iterable[str] = (),
        detach_dsm_taps: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._spmf_spec = SPMF_VARIANTS[self.spmf_variant]
        for name in ("fusion1", "fusion2", "fusion3", "fusion4"):
            if hasattr(self, name):
                delattr(self, name)

        setattr(
            self,
            self._spmf_variant_spec.indexes_attr,
            _resolve_spmf_indexes(self.image_encoder, self._spmf_variant_spec.variant_name),
        )
        tap_channels = int(self.image_encoder.embed_dim)
        setattr(
            self,
            self._spmf_variant_spec.structure_attr,
            self._spmf_variant_spec.structure_branch_cls(
                tap_channels=tap_channels,
                output_channels=256,
                detach_dsm_taps=detach_dsm_taps,
            ),
        )
        setattr(
            self,
            self._spmf_variant_spec.fusion_attr,
            self._spmf_variant_spec.fusion_cls(
                channels=256,
                structure_channels=256,
                hidden_dim=64,
            ),
        )
        if record_intermediate_stats:
            attach_requested_intermediate_stats(
                self,
                record_intermediate_modules,
                {
                    module_name: [(getattr(self, attr_name), prefix)]
                    for module_name, attr_name, prefix in self._spmf_variant_spec.intermediate_modules
                },
            )

    def set_detach_dsm_taps(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError(f"Expected enabled to be a bool, got {type(enabled).__name__}.")
        structure_branch = getattr(self, self._spmf_variant_spec.structure_attr)
        structure_branch.detach_dsm_taps = enabled

    @property
    def _spmf_variant_spec(self) -> SPMFVariantSpec:
        return getattr(self, "_spmf_spec", SPMF_VARIANTS[self.spmf_variant])

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str = "Train") -> torch.Tensor:
        del mode
        h, w = x.size()[-2:]
        dsm = _as_single_channel_dsm(y)
        y_sam = self._make_sam_dsm_input(dsm)

        deepx, deepy, dsm_taps = self._encode_spmf_features(x, y_sam)
        rgb_feats = (
            self.fpn1x(deepx),
            self.fpn2x(deepx),
            self.fpn3x(deepx),
            self.fpn4x(deepx),
        )
        dsm_feats = (
            self.fpn1y(deepy),
            self.fpn2y(deepy),
            self.fpn3y(deepy),
            self.fpn4y(deepy),
        )
        structure_branch = getattr(self, self._spmf_variant_spec.structure_attr)
        spmf = getattr(self, self._spmf_variant_spec.fusion_attr)
        structure_feats = structure_branch(dsm, dsm_taps)
        res1, res2, res3, res4 = spmf(rgb_feats, dsm_feats, structure_feats)
        return self.decoder(res1, res2, res3, res4, h, w)

    def _make_sam_dsm_input(self, dsm: torch.Tensor) -> torch.Tensor:
        return dsm.repeat(1, 3, 1, 1)

    def _encode_spmf_features(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        encoder = self.image_encoder
        x = encoder.patch_embed(x)
        y = encoder.patch_embed(y)
        if encoder.pos_embed is not None:
            new_abs_pos = F.interpolate(
                encoder.pos_embed.permute(0, 3, 1, 2),
                size=(x.shape[1], x.shape[2]),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            x = x + new_abs_pos
            y = y + new_abs_pos

        dsm_taps: list[torch.Tensor] = []
        tap_indexes = set(getattr(self, self._spmf_variant_spec.indexes_attr))
        for index, block in enumerate(encoder.blocks):
            x, y = block(x, y)
            if index in tap_indexes:
                dsm_taps.append(y.permute(0, 3, 1, 2).contiguous())

        self._validate_spmf_taps(dsm_taps)
        deepx = encoder.neck(x.permute(0, 3, 1, 2))
        deepy = encoder.neck(y.permute(0, 3, 1, 2))
        tap1, tap2, tap3, tap4 = dsm_taps
        return deepx, deepy, (tap1, tap2, tap3, tap4)

    def _validate_spmf_taps(self, dsm_taps: Sequence[torch.Tensor]) -> None:
        if len(dsm_taps) != 4:
            raise ValueError(
                f"Expected exactly 4 DSM SAM taps for SPMF{self._spmf_variant_spec.variant_name}, got {len(dsm_taps)}."
            )


__all__ = [
    "SPMFVariantSpec",
    "SPMF_VARIANTS",
    "UNetFormerSPMF",
    "_as_single_channel_dsm",
    "_resolve_spmf_indexes",
]
