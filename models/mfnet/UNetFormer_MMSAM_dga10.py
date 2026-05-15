from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import IntermediateStatsRecorder

from .UNetFormer_MMSAM import UNetFormer
from .modules import DGABlock10


class UNetFormerDGA10(UNetFormer):
    def __init__(self, *args: object, record_intermediate_stats: bool = False, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.dga_indexes = self._resolve_dga_indexes()
        self.dga_blocks = nn.ModuleList(
            [DGABlock10(channels=int(self.image_encoder.embed_dim)) for _ in self.dga_indexes]
        )
        if record_intermediate_stats:
            self.intermediate_stats = IntermediateStatsRecorder()
            self._attach_intermediate_stats_to_dga_blocks()

    def _attach_intermediate_stats_to_dga_blocks(self) -> None:
        for index, block in enumerate(self.dga_blocks):
            block.intermediate_stats = self.intermediate_stats
            block.intermediate_stats_prefix = f"dga/block_{index}"

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str = "Train") -> torch.Tensor:
        del mode
        h, w = x.size()[-2:]
        y = torch.unsqueeze(y, dim=1).repeat(1, 3, 1, 1)
        deepx, deepy = self._encode_with_dga(x, y)

        res1x = self.fpn1x(deepx)
        res2x = self.fpn2x(deepx)
        res3x = self.fpn3x(deepx)
        res4x = self.fpn4x(deepx)
        res1y = self.fpn1y(deepy)
        res2y = self.fpn2y(deepy)
        res3y = self.fpn3y(deepy)
        res4y = self.fpn4y(deepy)

        res1 = self.fusion1(res1x, res1y)
        res2 = self.fusion2(res2x, res2y)
        res3 = self.fusion3(res3x, res3y)
        res4 = self.fusion4(res4x, res4y)
        return self.decoder(res1, res2, res3, res4, h, w)

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

    def _encode_with_dga(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        dga_by_index = dict(zip(self.dga_indexes, self.dga_blocks))
        for index, block in enumerate(encoder.blocks):
            x, y = block(x, y)
            dga = dga_by_index.get(index)
            if dga is not None:
                x_bchw, y_bchw = dga(x.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2))
                x = x_bchw.permute(0, 2, 3, 1)
                y = y_bchw.permute(0, 2, 3, 1)

        deepx = encoder.neck(x.permute(0, 3, 1, 2))
        deepy = encoder.neck(y.permute(0, 3, 1, 2))
        return deepx, deepy
