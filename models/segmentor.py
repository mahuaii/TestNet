from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .backbones import SimpleRGBBackbone
from .decode_heads import SimpleSegHead


class RGBSegmentor(nn.Module):
    """Minimal single-modal segmentor for RGB-only pipeline validation."""

    RGB_KEY = "rgb"
    NUM_CLASSES = 1

    def __init__(
        self,
        backbone: nn.Module | None = None,
        decode_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.rgb_key = self.RGB_KEY
        self.num_classes = self.NUM_CLASSES
        self.backbone = backbone or SimpleRGBBackbone()
        self.decode_head = decode_head or SimpleSegHead()

    def forward(self, inputs: dict[str, torch.Tensor], mode: str = "tensor") -> dict[str, Any]:
        if self.rgb_key not in inputs:
            raise KeyError(
                f"RGBSegmentor expected inputs['{self.rgb_key}'], "
                f"but got keys: {sorted(inputs.keys())}"
            )

        rgb = inputs[self.rgb_key]
        if rgb.ndim != 4:
            raise ValueError(
                f"Expected '{self.rgb_key}' to have shape [B, C, H, W], got {tuple(rgb.shape)}"
            )

        features = self.backbone(rgb)
        fused_feats = features["fused_feats"]
        last_feat = fused_feats[-1]
        seg_logits = self.decode_head(last_feat, output_size=rgb.shape[-2:])

        outputs: dict[str, Any] = {
            "seg_logits": seg_logits,
            "features": features,
        }

        if mode == "predict":
            if seg_logits.shape[1] == 1:
                outputs["pred_mask"] = (torch.sigmoid(seg_logits) > 0.5).long().squeeze(1)
            else:
                outputs["pred_mask"] = torch.argmax(seg_logits, dim=1)
            return outputs
        if mode in {"tensor", "loss"}:
            return outputs

        raise ValueError(f"Unsupported mode: {mode}")
