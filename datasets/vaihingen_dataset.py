from __future__ import annotations

from collections.abc import Sequence
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset

VAIHINGEN_TRAIN_IDS = ["1", "3", "23", "26", "7", "11", "13", "28", "17", "32", "34", "37"]
VAIHINGEN_VAL_IDS = ["5", "21", "15", "30"]
VAIHINGEN_PALETTE = {
    0: (255, 255, 255),
    1: (0, 0, 255),
    2: (0, 255, 255),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (255, 0, 0),
    6: (0, 0, 0),
}
VAIHINGEN_INVERT_PALETTE = {value: key for key, value in VAIHINGEN_PALETTE.items()}


def convert_from_color(arr_3d: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr_3d)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected label image with shape [H, W, 3], got {tuple(arr.shape)}")

    arr_2d = np.zeros(arr.shape[:2], dtype=np.uint8)
    for color, label in VAIHINGEN_INVERT_PALETTE.items():
        mask = np.all(arr == np.asarray(color, dtype=arr.dtype).reshape(1, 1, 3), axis=2)
        arr_2d[mask] = label
    return arr_2d


def _normalize_dsm(dsm: np.ndarray) -> np.ndarray:
    dsm = np.asarray(dsm, dtype=np.float32)
    dsm_min = float(np.min(dsm))
    dsm_max = float(np.max(dsm))
    return (dsm - dsm_min) / (dsm_max - dsm_min + 1e-8)


class VaihingenDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        ids: Sequence[str],
        patch_size: Sequence[int] = (256, 256),
        samples_per_epoch: int | None = None,
        cache: bool = True,
        augmentation: bool = True,
        split: str = "train",
    ) -> None:
        if not ids:
            raise ValueError("VaihingenDataset requires at least one tile id")

        self.root_dir = Path(root_dir)
        self.ids = [str(item) for item in ids]
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.samples_per_epoch = int(samples_per_epoch or len(self.ids))
        self.cache = bool(cache)
        self.augmentation = bool(augmentation)
        self.split = str(split)

        self.rgb_files = [self.root_dir / "rgb" / f"top_mosaic_09cm_area{tile_id}.tif" for tile_id in self.ids]
        self.dsm_files = [
            self.root_dir / "dsm" / f"dsm_09cm_matching_area{tile_id}.tif" for tile_id in self.ids
        ]
        self.label_files = [
            self.root_dir / "labels" / f"top_mosaic_09cm_area{tile_id}.tif" for tile_id in self.ids
        ]

        for path in [*self.rgb_files, *self.dsm_files, *self.label_files]:
            if not path.is_file():
                raise FileNotFoundError(f"VaihingenDataset expected file at {path}")

        self.rgb_cache: dict[int, np.ndarray] = {}
        self.dsm_cache: dict[int, np.ndarray] = {}
        self.label_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, object]:
        tile_index = random.randrange(len(self.ids)) if self.split == "train" else index % len(self.ids)
        rgb = self._load_rgb(tile_index)
        dsm = self._load_dsm(tile_index)
        target = self._load_target(tile_index)

        rgb_patch, dsm_patch, target_patch = self._crop_patch(rgb, dsm, target)
        if self.augmentation:
            rgb_patch, dsm_patch, target_patch = self._augment(rgb_patch, dsm_patch, target_patch)

        return {
            "inputs": {
                "rgb": torch.from_numpy(rgb_patch.copy()).float(),
                "dsm": torch.from_numpy(dsm_patch.copy()).float(),
            },
            "target": torch.from_numpy(target_patch.copy()).long(),
            "meta": {
                "sample_index": index,
                "tile_id": self.ids[tile_index],
            },
        }

    def _load_rgb(self, tile_index: int) -> np.ndarray:
        if tile_index not in self.rgb_cache:
            rgb = np.asarray(imageio.imread(self.rgb_files[tile_index]), dtype=np.float32)
            if rgb.ndim != 3 or rgb.shape[2] < 3:
                raise ValueError(
                    f"Expected RGB tile with at least 3 channels, got {tuple(rgb.shape)} "
                    f"from {self.rgb_files[tile_index]}"
                )
            rgb = (rgb[:, :, :3] / 255.0).transpose(2, 0, 1)
            if self.cache:
                self.rgb_cache[tile_index] = rgb
        return self.rgb_cache[tile_index] if self.cache else (
            np.asarray(imageio.imread(self.rgb_files[tile_index]), dtype=np.float32)[:, :, :3] / 255.0
        ).transpose(2, 0, 1)

    def _load_dsm(self, tile_index: int) -> np.ndarray:
        if tile_index not in self.dsm_cache:
            dsm = _normalize_dsm(imageio.imread(self.dsm_files[tile_index]))
            if self.cache:
                self.dsm_cache[tile_index] = dsm
        return self.dsm_cache[tile_index] if self.cache else _normalize_dsm(
            imageio.imread(self.dsm_files[tile_index])
        )

    def _load_target(self, tile_index: int) -> np.ndarray:
        if tile_index not in self.label_cache:
            target = convert_from_color(np.asarray(imageio.imread(self.label_files[tile_index])))
            if self.cache:
                self.label_cache[tile_index] = target
        return self.label_cache[tile_index] if self.cache else convert_from_color(
            np.asarray(imageio.imread(self.label_files[tile_index]))
        )

    def _crop_patch(
        self, rgb: np.ndarray, dsm: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        patch_h, patch_w = self.patch_size
        height, width = rgb.shape[-2:]
        if patch_h > height or patch_w > width:
            raise ValueError(
                f"Patch size {self.patch_size} exceeds tile size {(height, width)} for split {self.split!r}"
            )

        if self.split == "train":
            x1 = random.randint(0, height - patch_h)
            y1 = random.randint(0, width - patch_w)
        else:
            x1 = (height - patch_h) // 2
            y1 = (width - patch_w) // 2

        x2 = x1 + patch_h
        y2 = y1 + patch_w
        return rgb[:, x1:x2, y1:y2], dsm[x1:x2, y1:y2], target[x1:x2, y1:y2]

    def _augment(
        self, rgb: np.ndarray, dsm: np.ndarray, target: np.ndarray
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
