from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F

from .UNetFormer_MMSAM_dga20 import UNetFormerDGA20
from .intermediate_stats_config import attach_requested_intermediate_stats
from .modules import DGSF10


class UNetFormerDGA20DGSF10(UNetFormerDGA20):
    def __init__(
        self,
        *args: object,
        record_intermediate_stats: bool = False,
        record_intermediate_modules: Iterable[str] = (),
        **kwargs: object,
    ) -> None:
        super().__init__(
            *args,
            record_intermediate_stats=False,
            record_intermediate_modules=(),
            **kwargs,
        )
        self.dgsf10 = DGSF10(input_channels=int(self.image_encoder.embed_dim), hidden_channels=256)
        if record_intermediate_stats:
            attach_requested_intermediate_stats(
                self,
                record_intermediate_modules,
                {
                    "dga": [
                        (block, f"dga/block_{index}")
                        for index, block in enumerate(self.dga_blocks)
                    ],
                    "dgsf10": [(self.dgsf10, "dgsf10")],
                },
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str = "Train") -> torch.Tensor:
        del mode
        h, w = x.size()[-2:]
        y = torch.unsqueeze(y, dim=1).repeat(1, 3, 1, 1)
        rgb_feats, aux_feats = self._encode_dgsf10_features(x, y)
        res1, res2, res3, res4 = self.dgsf10(rgb_feats, aux_feats)
        return self.decoder(res1, res2, res3, res4, h, w)

    def _encode_dgsf10_features(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
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

        rgb_dga_features: list[torch.Tensor] = []
        aux_dga_features: list[torch.Tensor] = []
        dga_by_index = dict(zip(self.dga_indexes, self.dga_blocks))
        for index, block in enumerate(encoder.blocks):
            x, y = block(x, y)
            dga = dga_by_index.get(index)
            if dga is not None:
                x_bchw, y_bchw = dga(x.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2))
                rgb_dga_features.append(x_bchw)
                aux_dga_features.append(y_bchw)
                x = x_bchw.permute(0, 2, 3, 1)
                y = y_bchw.permute(0, 2, 3, 1)

        self._validate_dgsf10_dga_features(rgb_dga_features, aux_dga_features)
        rgb_top = x.permute(0, 3, 1, 2)
        aux_top = y.permute(0, 3, 1, 2)
        return (*rgb_dga_features, rgb_top), (*aux_dga_features, aux_top)

    @staticmethod
    def _validate_dgsf10_dga_features(
        rgb_feats: Sequence[torch.Tensor],
        aux_feats: Sequence[torch.Tensor],
    ) -> None:
        if len(rgb_feats) != 4 or len(aux_feats) != 4:
            raise ValueError(
                "Expected exactly 4 DGA feature pairs for DGSF10, "
                f"got {len(rgb_feats)} RGB and {len(aux_feats)} aux features."
            )


__all__ = ["UNetFormerDGA20DGSF10"]
