from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F


class DataUtils:
    @staticmethod
    def convert_from_color(
        arr_3d: np.ndarray,
        invert_palette: dict[tuple[int, int, int], int],
    ) -> np.ndarray:
        arr = np.asarray(arr_3d)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected label image with shape [H, W, 3], got {tuple(arr.shape)}")

        arr_2d = np.zeros(arr.shape[:2], dtype=np.uint8)
        for color, label in invert_palette.items():
            mask = np.all(arr == np.asarray(color, dtype=arr.dtype).reshape(1, 1, 3), axis=2)
            arr_2d[mask] = label
        return arr_2d

    @staticmethod
    def normalize_dsm(dsm: np.ndarray) -> np.ndarray:
        dsm = np.asarray(dsm, dtype=np.float32)
        dsm_min = float(np.min(dsm))
        dsm_max = float(np.max(dsm))
        return (dsm - dsm_min) / (dsm_max - dsm_min + 1e-8)

    @staticmethod
    def augment_triplet(
        rgb: np.ndarray,
        dsm: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            rgb = rgb[:, ::-1, :]
            dsm = dsm[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:
            rgb = rgb[:, :, ::-1]
            dsm = dsm[:, ::-1]
            target = target[:, ::-1]
        return rgb.copy(), dsm.copy(), target.copy()

    @staticmethod
    def cross_entropy_filtered(
        logits: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        ignore_label: int = 255,
    ) -> torch.Tensor:
        """
        - logits: [B, C, H, W]
        - target: [B, H, W] with values in [0, C-1] or ignore_label
        - weight: [C] or None
        - ignore_label: int
        """
        if logits.ndim != 4:
            raise ValueError(f"Expected logits with shape [B, C, H, W], got {tuple(logits.shape)}")

        n, c, h, w = logits.shape
        if target.shape != (n, h, w):
            raise ValueError(
                "Expected target shape to match logits spatial dims, "
                f"got logits {tuple(logits.shape)} and target {tuple(target.shape)}"
            )

        target_mask = (target >= 0) & (target != ignore_label)
        if not torch.any(target_mask):
            return logits.sum() * 0.0

        filtered_target = target[target_mask]
        filtered_logits = logits.permute(0, 2, 3, 1).contiguous()
        filtered_logits = filtered_logits[target_mask].view(-1, c)
        return F.cross_entropy(
            filtered_logits,
            filtered_target,
            weight=weight,
        )
