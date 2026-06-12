from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .UNetFormer_MMSAM import UNetFormer
from .intermediate_stats_config import attach_requested_intermediate_stats
from .modules import DSMStructureBranch10, MultiScaleStructurePriorModulatedFusion20


def _resolve_spmf20_indexes(image_encoder: nn.Module) -> list[int]:
    global_indexes = [
        index for index, block in enumerate(image_encoder.blocks) if getattr(block, "window_size", None) == 0
    ]
    if len(global_indexes) != 4:
        raise ValueError(
            "Expected exactly 4 encoder feature taps from global attention blocks for SPMF20, "
            f"got {len(global_indexes)}: {global_indexes}."
        )
    return [int(index) for index in global_indexes]


def _as_single_channel_dsm(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 3:
        return y.unsqueeze(1)
    if y.ndim == 4 and y.shape[1] == 1:
        return y
    raise ValueError(f"Expected DSM with shape [B, H, W] or [B, 1, H, W], got {tuple(y.shape)}.")


class UNetFormerSPMF20(UNetFormer):
    def __init__(
        self,
        *args: object,
        record_intermediate_stats: bool = False,
        record_intermediate_modules: Iterable[str] = (),
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        for name in ("fusion1", "fusion2", "fusion3", "fusion4"):
            if hasattr(self, name):
                delattr(self, name)

        self.spmf20_indexes = _resolve_spmf20_indexes(self.image_encoder)
        tap_channels = int(self.image_encoder.embed_dim)
        self.structure_branch10 = DSMStructureBranch10(tap_channels=tap_channels, output_channels=256)
        self.spmf20 = MultiScaleStructurePriorModulatedFusion20(
            channels=256,
            structure_channels=256,
            hidden_dim=64,
        )
        if record_intermediate_stats:
            attach_requested_intermediate_stats(
                self,
                record_intermediate_modules,
                {
                    "spmf20": [(self.spmf20, "spmf20")],
                    "structure10": [(self.structure_branch10, "spmf20/structure")],
                },
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str = "Train") -> torch.Tensor:
        del mode
        h, w = x.size()[-2:]
        dsm = _as_single_channel_dsm(y)
        y_sam = dsm.repeat(1, 3, 1, 1)

        deepx, deepy, dsm_taps = self._encode_spmf20_features(x, y_sam)
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
        structure_feats = self.structure_branch10(dsm, dsm_taps)
        res1, res2, res3, res4 = self.spmf20(rgb_feats, dsm_feats, structure_feats)
        return self.decoder(res1, res2, res3, res4, h, w)

    def _encode_spmf20_features(
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
        tap_indexes = set(self.spmf20_indexes)
        for index, block in enumerate(encoder.blocks):
            x, y = block(x, y)
            if index in tap_indexes:
                dsm_taps.append(y.permute(0, 3, 1, 2).contiguous())

        self._validate_spmf20_taps(dsm_taps)
        deepx = encoder.neck(x.permute(0, 3, 1, 2))
        deepy = encoder.neck(y.permute(0, 3, 1, 2))
        tap1, tap2, tap3, tap4 = dsm_taps
        return deepx, deepy, (tap1, tap2, tap3, tap4)

    @staticmethod
    def _validate_spmf20_taps(dsm_taps: Sequence[torch.Tensor]) -> None:
        if len(dsm_taps) != 4:
            raise ValueError(f"Expected exactly 4 DSM SAM taps for SPMF20, got {len(dsm_taps)}.")


__all__ = ["UNetFormerSPMF20"]
