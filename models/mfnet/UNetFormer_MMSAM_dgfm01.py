from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .UNetFormer_MMSAM import UNetFormer
from .UNetFormer_MMSAM_dgfm import _resolve_dgfm_indexes
from .intermediate_stats_config import attach_requested_intermediate_stats
from .modules import DGFM01
from .modules.scale_adapter import DGFMScaleAdapter


def _init_dgfm01_modules(
    model: nn.Module,
    record_intermediate_stats: bool,
    record_intermediate_modules: Iterable[str],
) -> None:
    model.dgfm01_indexes = _resolve_dgfm_indexes(model.image_encoder)  # type: ignore[attr-defined]
    embed_dim = int(model.image_encoder.embed_dim)  # type: ignore[attr-defined]
    model.dgfm01_input_norms = nn.ModuleList(  # type: ignore[attr-defined]
        [nn.LayerNorm(embed_dim) for _ in model.dgfm01_indexes]  # type: ignore[attr-defined]
    )
    model.dgfm01_blocks = nn.ModuleList(  # type: ignore[attr-defined]
        [DGFM01(dims=embed_dim) for _ in model.dgfm01_indexes]  # type: ignore[attr-defined]
    )
    model.dgfm01_output_norms = nn.ModuleList(  # type: ignore[attr-defined]
        [nn.LayerNorm(embed_dim) for _ in model.dgfm01_indexes]  # type: ignore[attr-defined]
    )
    model.dgfm01_output_projs = nn.ModuleList(  # type: ignore[attr-defined]
        [nn.Conv2d(embed_dim, 256, kernel_size=1, bias=False) for _ in model.dgfm01_indexes]  # type: ignore[attr-defined]
    )
    model.dgfm01_scale_adapters = nn.ModuleList(  # type: ignore[attr-defined]
        [DGFMScaleAdapter(256) for _ in model.dgfm01_indexes]  # type: ignore[attr-defined]
    )
    if record_intermediate_stats:
        attach_requested_intermediate_stats(
            model,
            record_intermediate_modules,
            {
                "dgfm01": [
                    (block, f"dgfm01/block_{index}")
                    for index, block in enumerate(model.dgfm01_blocks)  # type: ignore[attr-defined]
                ]
            },
        )


def _encode_dgfm01_decoder_features(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder = model.image_encoder  # type: ignore[attr-defined]
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

    dgfm01_by_index = {
        index: tap_index
        for tap_index, index in enumerate(model.dgfm01_indexes)  # type: ignore[attr-defined]
    }
    decoder_features: list[torch.Tensor | None] = [None, None, None, None]
    for index, block in enumerate(encoder.blocks):
        x, y = block(x, y)
        tap_index = dgfm01_by_index.get(index)
        if tap_index is None:
            continue

        input_norm = model.dgfm01_input_norms[tap_index]  # type: ignore[attr-defined]
        dgfm01 = model.dgfm01_blocks[tap_index]  # type: ignore[attr-defined]
        output_norm = model.dgfm01_output_norms[tap_index]  # type: ignore[attr-defined]
        output_proj = model.dgfm01_output_projs[tap_index]  # type: ignore[attr-defined]
        scale_adapter = model.dgfm01_scale_adapters[tap_index]  # type: ignore[attr-defined]

        fused = dgfm01(input_norm(x), input_norm(y))
        fused = output_norm(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        dgfm01_outputs = scale_adapter(output_proj(fused))
        if len(dgfm01_outputs) != 4:
            raise ValueError(f"Expected DGFM01 scale adapter to return 4 decoder features, got {len(dgfm01_outputs)}.")
        decoder_features[tap_index] = dgfm01_outputs[tap_index]

    missing_indexes = [index for index, feature in enumerate(decoder_features) if feature is None]
    if missing_indexes:
        raise ValueError(f"Missing DGFM01 decoder features for tap indexes {missing_indexes}.")

    res1, res2, res3, res4 = decoder_features
    assert res1 is not None and res2 is not None and res3 is not None and res4 is not None
    return res1, res2, res3, res4


class UNetFormerDGFM01(UNetFormer):
    def __init__(
        self,
        *args: object,
        record_intermediate_stats: bool = False,
        record_intermediate_modules: Iterable[str] = (),
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        _init_dgfm01_modules(self, record_intermediate_stats, record_intermediate_modules)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str = "Train") -> torch.Tensor:
        del mode
        h, w = x.size()[-2:]
        y = torch.unsqueeze(y, dim=1).repeat(1, 3, 1, 1)
        res1, res2, res3, res4 = _encode_dgfm01_decoder_features(self, x, y)
        return self.decoder(res1, res2, res3, res4, h, w)


__all__ = ["UNetFormerDGFM01"]
