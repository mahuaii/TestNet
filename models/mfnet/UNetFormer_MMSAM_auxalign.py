from __future__ import annotations

import torch
import torch.nn.functional as F

from .UNetFormer_MMSAM import UNetFormer


class UNetFormerAuxAlign(UNetFormer):
    def __init__(
        self,
        *args: object,
        align_index: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.align_index = self._resolve_align_index(align_index)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mode: str = "Train",
        return_align: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del mode
        h, w = x.size()[-2:]
        y = torch.unsqueeze(y, dim=1).repeat(1, 3, 1, 1)
        if return_align:
            deepx, deepy, x_align_feat, y_align_feat = self._encode_with_align(x, y)
            logits = self._decode_from_deep_features(deepx, deepy, h, w)
            return logits, x_align_feat, y_align_feat

        deepx, deepy = self.image_encoder(x, y)
        return self._decode_from_deep_features(deepx, deepy, h, w)

    def _resolve_align_index(self, align_index: int | None) -> int:
        if align_index is not None:
            return int(align_index)

        global_indexes = [
            index
            for index, block in enumerate(self.image_encoder.blocks)
            if getattr(block, "window_size", None) == 0
        ]
        if not global_indexes:
            raise ValueError("Could not infer an align block: no global attention blocks were found.")

        inferred_index = int(global_indexes[0])
        if len(self.image_encoder.blocks) == 12 and inferred_index != 2:
            raise ValueError(
                "Expected SAM ViT-B first global attention block at index 2, "
                f"but inferred index {inferred_index}."
            )
        return inferred_index

    def _encode_with_align(
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
        for index, block in enumerate(encoder.blocks):
            x, y = block(x, y)
            if index == self.align_index:
                x_align_feat = x
                y_align_feat = y

        if x_align_feat is None or y_align_feat is None:
            raise ValueError(
                f"align_index {self.align_index} is outside the encoder block range "
                f"[0, {len(encoder.blocks) - 1}]."
            )

        deepx = encoder.neck(x.permute(0, 3, 1, 2))
        deepy = encoder.neck(y.permute(0, 3, 1, 2))
        return deepx, deepy, x_align_feat, y_align_feat

    def _decode_from_deep_features(
        self,
        deepx: torch.Tensor,
        deepy: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
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


__all__ = ["UNetFormerAuxAlign"]
