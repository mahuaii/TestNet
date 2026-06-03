from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .UNetFormer_MMSAM import UNetFormer
from .intermediate_stats_config import attach_requested_intermediate_stats
from .modules import DGFM


def _resolve_dgfm_indexes(image_encoder: nn.Module) -> list[int]:
    blocks = image_encoder.blocks
    global_indexes = [
        index for index, block in enumerate(blocks) if getattr(block, "window_size", None) == 0
    ]
    if len(global_indexes) < 3:
        raise ValueError(
            "Expected at least 3 global attention blocks for DGFM taps, "
            f"got {len(global_indexes)}: {global_indexes}."
        )

    indexes = [*global_indexes[:3], len(blocks) - 1]
    if len(set(indexes)) != 4:
        raise ValueError(
            "Expected DGFM taps from the first 3 global attention blocks plus a distinct deepest block, "
            f"got indexes {indexes} from global attention blocks {global_indexes}."
        )
    return [int(index) for index in indexes]


def _init_dgfm_modules(
    model: nn.Module,
    record_intermediate_stats: bool,
    record_intermediate_modules: Iterable[str],
) -> None:
    model.dgfm_indexes = _resolve_dgfm_indexes(model.image_encoder)  # type: ignore[attr-defined]
    model.dgfm_blocks = nn.ModuleList(  # type: ignore[attr-defined]
        [
            DGFM(dims=int(model.image_encoder.embed_dim), out_channels=256)  # type: ignore[attr-defined]
            for _ in model.dgfm_indexes  # type: ignore[attr-defined]
        ]
    )
    if record_intermediate_stats:
        attach_requested_intermediate_stats(
            model,
            record_intermediate_modules,
            {
                "dgfm": [
                    (block, f"dgfm/block_{index}")
                    for index, block in enumerate(model.dgfm_blocks)  # type: ignore[attr-defined]
                ]
            },
        )


def _encode_dgfm_decoder_features(
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

    dgfm_indexes = model.dgfm_indexes  # type: ignore[attr-defined]
    dgfm_blocks = model.dgfm_blocks  # type: ignore[attr-defined]
    dgfm_by_index = {
        index: (tap_index, dgfm)
        for tap_index, (index, dgfm) in enumerate(zip(dgfm_indexes, dgfm_blocks))
    }
    decoder_features: list[torch.Tensor | None] = [None, None, None, None]
    for index, block in enumerate(encoder.blocks):
        x, y = block(x, y)
        dgfm_entry = dgfm_by_index.get(index)
        if dgfm_entry is None:
            continue
        tap_index, dgfm = dgfm_entry
        dgfm_outputs = dgfm(x, y)
        if len(dgfm_outputs) != 4:
            raise ValueError(f"Expected DGFM to return 4 decoder features, got {len(dgfm_outputs)}.")
        decoder_features[tap_index] = dgfm_outputs[tap_index]

    missing_indexes = [index for index, feature in enumerate(decoder_features) if feature is None]
    if missing_indexes:
        raise ValueError(f"Missing DGFM decoder features for tap indexes {missing_indexes}.")

    res1, res2, res3, res4 = decoder_features
    assert res1 is not None and res2 is not None and res3 is not None and res4 is not None
    return res1, res2, res3, res4


class UNetFormerDGFM(UNetFormer):
    def __init__(
        self,
        *args: object,
        record_intermediate_stats: bool = False,
        record_intermediate_modules: Iterable[str] = (),
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        _init_dgfm_modules(self, record_intermediate_stats, record_intermediate_modules)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str = "Train") -> torch.Tensor:
        del mode
        h, w = x.size()[-2:]
        y = torch.unsqueeze(y, dim=1).repeat(1, 3, 1, 1)
        res1, res2, res3, res4 = _encode_dgfm_decoder_features(self, x, y)
        return self.decoder(res1, res2, res3, res4, h, w)


__all__ = ["UNetFormerDGFM"]
