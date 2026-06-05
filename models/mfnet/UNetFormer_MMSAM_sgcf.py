from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .UNetFormer_MMSAM import UNetFormer
from .intermediate_stats_config import attach_requested_intermediate_stats
from .modules import SGCF


def _resolve_sgcf_indexes(image_encoder: nn.Module) -> list[int]:
    blocks = image_encoder.blocks
    global_indexes = [
        index for index, block in enumerate(blocks) if getattr(block, "window_size", None) == 0
    ]
    if len(global_indexes) < 3:
        raise ValueError(
            "Expected at least 3 global attention blocks for SGCF taps, "
            f"got {len(global_indexes)}: {global_indexes}."
        )

    indexes = [*global_indexes[:3], len(blocks) - 1]
    if len(set(indexes)) != 4:
        raise ValueError(
            "Expected SGCF taps from the first 3 global attention blocks plus a distinct deepest block, "
            f"got indexes {indexes} from global attention blocks {global_indexes}."
        )
    return [int(index) for index in indexes]


def _init_sgcf_modules(
    model: nn.Module,
    record_intermediate_stats: bool,
    record_intermediate_modules: Iterable[str],
) -> None:
    model.sgcf_indexes = _resolve_sgcf_indexes(model.image_encoder)  # type: ignore[attr-defined]
    model.sgcf_blocks = nn.ModuleList(  # type: ignore[attr-defined]
        [
            SGCF(dims=int(model.image_encoder.embed_dim), out_channels=256)  # type: ignore[attr-defined]
            for _ in model.sgcf_indexes  # type: ignore[attr-defined]
        ]
    )
    if record_intermediate_stats:
        attach_requested_intermediate_stats(
            model,
            record_intermediate_modules,
            {
                "sgcf": [
                    (block, f"sgcf/block_{index}")
                    for index, block in enumerate(model.sgcf_blocks)  # type: ignore[attr-defined]
                ]
            },
        )


def _encode_sgcf_decoder_features(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    dsm: torch.Tensor,
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

    sgcf_indexes = model.sgcf_indexes  # type: ignore[attr-defined]
    sgcf_blocks = model.sgcf_blocks  # type: ignore[attr-defined]
    sgcf_by_index = {
        index: (tap_index, sgcf)
        for tap_index, (index, sgcf) in enumerate(zip(sgcf_indexes, sgcf_blocks))
    }
    decoder_features: list[torch.Tensor | None] = [None, None, None, None]
    for index, block in enumerate(encoder.blocks):
        x, y = block(x, y)
        sgcf_entry = sgcf_by_index.get(index)
        if sgcf_entry is None:
            continue
        tap_index, sgcf = sgcf_entry
        sgcf_outputs = sgcf(x, y, dsm)
        if len(sgcf_outputs) != 4:
            raise ValueError(f"Expected SGCF to return 4 decoder features, got {len(sgcf_outputs)}.")
        decoder_features[tap_index] = sgcf_outputs[tap_index]

    missing_indexes = [index for index, feature in enumerate(decoder_features) if feature is None]
    if missing_indexes:
        raise ValueError(f"Missing SGCF decoder features for tap indexes {missing_indexes}.")

    res1, res2, res3, res4 = decoder_features
    assert res1 is not None and res2 is not None and res3 is not None and res4 is not None
    return res1, res2, res3, res4


class UNetFormerSGCF(UNetFormer):
    def __init__(
        self,
        *args: object,
        record_intermediate_stats: bool = False,
        record_intermediate_modules: Iterable[str] = (),
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        _init_sgcf_modules(self, record_intermediate_stats, record_intermediate_modules)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mode: str = "Train") -> torch.Tensor:
        del mode
        h, w = x.size()[-2:]
        if y.ndim == 3:
            dsm = y.unsqueeze(1)
        else:
            dsm = y
        y = dsm.repeat(1, 3, 1, 1)
        res1, res2, res3, res4 = _encode_sgcf_decoder_features(self, x, y, dsm)
        return self.decoder(res1, res2, res3, res4, h, w)


__all__ = ["UNetFormerSGCF"]
