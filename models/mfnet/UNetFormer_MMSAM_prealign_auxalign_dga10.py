from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .UNetFormer_MMSAM_prealign_auxalign import UNetFormerPreAlignAuxAlign
from .modules import DGABlock10


class UNetFormerPreAlignAuxAlignDGA10(UNetFormerPreAlignAuxAlign):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.dga_indexes = self._resolve_dga_indexes()
        self.dga_blocks = nn.ModuleList(
            [DGABlock10(channels=int(self.image_encoder.embed_dim)) for _ in self.dga_indexes]
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
        deepx, deepy, x_align_feat, y_align_feat = self._encode_with_dga_and_align(x, y_aligned)
        logits = self._decode_from_deep_features(deepx, deepy, h, w)
        if return_align:
            return logits, x_align_feat, y_align_feat
        return logits

    def _resolve_dga_indexes(self) -> list[int]:
        global_indexes = [
            index for index, block in enumerate(self.image_encoder.blocks) if getattr(block, "window_size", None) == 0
        ]
        if len(global_indexes) != 4:
            raise ValueError(
                "Expected exactly 4 DGA insertion blocks from global attention blocks, "
                f"got {len(global_indexes)}: {global_indexes}."
            )
        return [int(index) for index in global_indexes]

    def _encode_with_dga_and_align(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        dga_by_index = dict(zip(self.dga_indexes, self.dga_blocks))
        for index, block in enumerate(encoder.blocks):
            x, y = block(x, y)
            if index == self.align_index:
                x_align_feat = x
                y_align_feat = y
            dga = dga_by_index.get(index)
            if dga is not None:
                x_bchw, y_bchw = dga(x.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2))
                x = x_bchw.permute(0, 2, 3, 1)
                y = y_bchw.permute(0, 2, 3, 1)

        if x_align_feat is None or y_align_feat is None:
            raise ValueError(
                f"align_index {self.align_index} is outside the encoder block range "
                f"[0, {len(encoder.blocks) - 1}]."
            )

        deepx = encoder.neck(x.permute(0, 3, 1, 2))
        deepy = encoder.neck(y.permute(0, 3, 1, 2))
        return deepx, deepy, x_align_feat, y_align_feat


__all__ = ["UNetFormerPreAlignAuxAlignDGA10"]
