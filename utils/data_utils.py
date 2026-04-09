from __future__ import annotations

import random

import numpy as np


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
