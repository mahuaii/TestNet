from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F

from .UNetFormer_MMSAM_prealign_auxalign import UNetFormerPreAlignAuxAlign
from .intermediate_stats_config import attach_requested_intermediate_stats
from .modules import DGSF10


class UNetFormerPreAlignAuxAlignDGSF10(UNetFormerPreAlignAuxAlign):
    def __init__(
        self,
        *args: object,
        record_intermediate_stats: bool = False,
        record_intermediate_modules: Iterable[str] = (),
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dgsf10_indexes = self._resolve_dgsf10_indexes()
        self.dgsf10 = DGSF10(input_channels=int(self.image_encoder.embed_dim), hidden_channels=256)
        if record_intermediate_stats:
            attach_requested_intermediate_stats(
                self,
                record_intermediate_modules,
                {"dgsf10": [(self.dgsf10, "dgsf10")]},
            )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mode: str = "Train",
        return_align: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del mode
        h, w = x.size()[-2:]
        if y.ndim == 3:
            y = y.unsqueeze(1)
        y_aligned = self.aux_prealign(y)
        rgb_feats, aux_feats, x_align_feat, y_align_feat = self._encode_dgsf10_features_with_align(x, y_aligned)
        res1, res2, res3, res4 = self.dgsf10(rgb_feats, aux_feats)
        logits = self.decoder(res1, res2, res3, res4, h, w)
        if return_align:
            return logits, x_align_feat, y_align_feat
        return logits

    def _resolve_dgsf10_indexes(self) -> list[int]:
        global_indexes = [
            index for index, block in enumerate(self.image_encoder.blocks) if getattr(block, "window_size", None) == 0
        ]
        if len(global_indexes) != 4:
            raise ValueError(
                "Expected exactly 4 encoder feature taps from global attention blocks for DGSF10, "
                f"got {len(global_indexes)}: {global_indexes}."
            )
        return [int(index) for index in global_indexes]

    def _encode_dgsf10_features_with_align(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
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

        x_align_feat: torch.Tensor | None = None
        y_align_feat: torch.Tensor | None = None
        rgb_encoder_features: list[torch.Tensor] = []
        aux_encoder_features: list[torch.Tensor] = []
        tap_indexes = set(self.dgsf10_indexes)
        for index, block in enumerate(encoder.blocks):
            x, y = block(x, y)
            if index == self.align_index:
                x_align_feat = x
                y_align_feat = y
            if index in tap_indexes:
                rgb_encoder_features.append(x.permute(0, 3, 1, 2))
                aux_encoder_features.append(y.permute(0, 3, 1, 2))

        if x_align_feat is None or y_align_feat is None:
            raise ValueError(
                f"align_index {self.align_index} is outside the encoder block range "
                f"[0, {len(encoder.blocks) - 1}]."
            )

        self._validate_dgsf10_encoder_features(rgb_encoder_features, aux_encoder_features)
        rgb_top = x.permute(0, 3, 1, 2)
        aux_top = y.permute(0, 3, 1, 2)
        return (*rgb_encoder_features, rgb_top), (*aux_encoder_features, aux_top), x_align_feat, y_align_feat

    @staticmethod
    def _validate_dgsf10_encoder_features(
        rgb_feats: Sequence[torch.Tensor],
        aux_feats: Sequence[torch.Tensor],
    ) -> None:
        if len(rgb_feats) != 4 or len(aux_feats) != 4:
            raise ValueError(
                "Expected exactly 4 encoder feature pairs for DGSF10, "
                f"got {len(rgb_feats)} RGB and {len(aux_feats)} aux features."
            )


__all__ = ["UNetFormerPreAlignAuxAlignDGSF10"]
