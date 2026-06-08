from __future__ import annotations

import random

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torch.autograd import Variable
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
        return (dsm - dsm_min) / (dsm_max - dsm_min)

    @staticmethod
    def enhance_dsm_similarity(
        dsm: np.ndarray,
        window_size: int = 7,
        sigma: float = 0.15,
        lambda_weight: float = 0.3,
        eps: float = 1e-6,
    ) -> np.ndarray:
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise ValueError(f"Expected window_size to be an integer, got {type(window_size).__name__}.")
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError(f"Expected window_size to be a positive odd integer, got {window_size}.")
        sigma = float(sigma)
        lambda_weight = float(lambda_weight)
        eps = float(eps)
        if sigma <= 0:
            raise ValueError(f"Expected sigma to be positive, got {sigma}.")

        dsm = np.asarray(dsm, dtype=np.float32)
        if dsm.ndim != 2:
            raise ValueError(f"Expected DSM with shape [H, W], got {tuple(dsm.shape)}.")

        dsm_min = float(np.min(dsm))
        dsm_max = float(np.max(dsm))
        denominator = dsm_max - dsm_min
        if denominator <= eps:
            return np.zeros_like(dsm, dtype=np.float32)

        normalized = (dsm - dsm_min) / denominator
        pad_width = window_size // 2
        padded = np.pad(normalized, pad_width=pad_width, mode="reflect")
        windows = sliding_window_view(padded, (window_size, window_size))
        centered_diff = windows - normalized[:, :, np.newaxis, np.newaxis]
        similarity = np.exp(-(centered_diff**2) / (2.0 * sigma**2))
        similarity_map = np.mean(similarity, axis=(-2, -1))
        enhanced = (normalized + lambda_weight * normalized * similarity_map) / (1.0 + lambda_weight)
        return np.clip(enhanced, 0.0, 1.0).astype(np.float32, copy=False)

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
            return Variable(torch.zeros(1))

        filtered_target = target[target_mask]
        filtered_logits = logits.permute(0, 2, 3, 1).contiguous()
        filtered_logits = filtered_logits[target_mask].view(-1, c)
        return F.cross_entropy(
            filtered_logits,
            filtered_target,
            weight=weight,
        )
