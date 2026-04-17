from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import DataUtils

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
POTSDAM_PALETTE = dict(VAIHINGEN_PALETTE)
POTSDAM_INVERT_PALETTE = {value: key for key, value in POTSDAM_PALETTE.items()}


@dataclass(frozen=True)
class ISPRSPreset:
    name: str
    rgb_subdir: str
    rgb_pattern: str
    dsm_subdir: str
    dsm_pattern: str
    label_subdir: str
    label_pattern: str
    invert_palette: dict[tuple[int, int, int], int]
    rgb_channels: tuple[int, ...] = (0, 1, 2)
    eval_label_subdir: str | None = None
    eval_label_pattern: str | None = None


VAIHINGEN_PRESET = ISPRSPreset(
    name="vaihingen",
    rgb_subdir="rgb",
    rgb_pattern="top_mosaic_09cm_area{tile_id}.tif",
    dsm_subdir="dsm",
    dsm_pattern="dsm_09cm_matching_area{tile_id}.tif",
    label_subdir="labels",
    label_pattern="top_mosaic_09cm_area{tile_id}.tif",
    invert_palette=VAIHINGEN_INVERT_PALETTE,
    eval_label_subdir="labels_eroded",
    eval_label_pattern="top_mosaic_09cm_area{tile_id}_noBoundary.tif",
)
POTSDAM_PRESET = ISPRSPreset(
    name="potsdam",
    rgb_subdir="rgbir",
    rgb_pattern="top_potsdam_{tile_id}_RGBIR.tif",
    dsm_subdir="dsm",
    dsm_pattern="dsm_potsdam_{tile_id}_normalized_lastools.jpg",
    label_subdir="labels",
    label_pattern="top_potsdam_{tile_id}_label.tif",
    invert_palette=POTSDAM_INVERT_PALETTE,
    eval_label_subdir="labels_eroded",
    eval_label_pattern="top_potsdam_{tile_id}_label_noBoundary.tif",
)


class ISPRSDataset(Dataset):
    def __init__(
        self,
        preset: ISPRSPreset,
        root_dir: str,
        ids: Sequence[str],
        patch_size: Sequence[int] = (256, 256),
        samples_per_epoch: int | None = None,
        cache: bool = True,
        augmentation: bool = True,
        split: str = "train",
    ) -> None:
        if not ids:
            raise ValueError(f"{preset.name} dataset requires at least one tile id")

        self.preset = preset
        self.root_dir = Path(root_dir)
        self.ids = [str(item) for item in ids]
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.samples_per_epoch = int(samples_per_epoch or len(self.ids))
        self.cache = bool(cache)
        self.augmentation = bool(augmentation)
        self.split = str(split)

        self.rgb_files = [
            self.root_dir / preset.rgb_subdir / preset.rgb_pattern.format(tile_id=tile_id)
            for tile_id in self.ids
        ]
        self.dsm_files = [
            self.root_dir / preset.dsm_subdir / preset.dsm_pattern.format(tile_id=tile_id)
            for tile_id in self.ids
        ]
        self.label_files = [
            self.root_dir / preset.label_subdir / preset.label_pattern.format(tile_id=tile_id)
            for tile_id in self.ids
        ]
        self.eval_label_files = [
            self.root_dir
            / (preset.eval_label_subdir or preset.label_subdir)
            / (preset.eval_label_pattern or preset.label_pattern).format(tile_id=tile_id)
            for tile_id in self.ids
        ]

        for path in [*self.rgb_files, *self.dsm_files, *self.label_files]:
            if not path.is_file():
                raise FileNotFoundError(f"{preset.name} dataset expected file at {path}")

        self.rgb_cache: dict[int, np.ndarray] = {}
        self.dsm_cache: dict[int, np.ndarray] = {}
        self.label_cache: dict[int, np.ndarray] = {}
        self.eval_label_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, object]:
        tile_index = self._resolve_tile_index(index)
        rgb = self._load_rgb_tile(tile_index)
        dsm = self._load_dsm_tile(tile_index)
        target = self._load_label_tile(
            tile_index=tile_index,
            files=self.label_files,
            cache_store=self.label_cache,
        )

        rgb_patch, dsm_patch, target_patch = self._crop_random_patch(rgb, dsm, target)
        if self.augmentation:
            rgb_patch, dsm_patch, target_patch = DataUtils.augment_triplet(
                rgb=rgb_patch,
                dsm=dsm_patch,
                target=target_patch,
            )

        sample = {
            "inputs": {
                "rgb": torch.from_numpy(rgb_patch.copy()).float(),
                "dsm": torch.from_numpy(dsm_patch.copy()).float(),
            },
            "target": torch.from_numpy(target_patch.copy()).long(),
            "meta": {
                "sample_index": index,
                "source_tile_id": self.ids[tile_index],
            },
        }
        return sample

    def get_tile(self, index: int) -> dict[str, object]:
        tile_index = int(index)
        rgb = self._load_rgb_tile(tile_index)
        dsm = self._load_dsm_tile(tile_index)
        target = self._load_label_tile(
            tile_index=tile_index,
            files=self.eval_label_files,
            cache_store=self.eval_label_cache,
        )
        return {
            "inputs": {
                "rgb": torch.from_numpy(rgb.copy()).float(),
                "dsm": torch.from_numpy(dsm.copy()).float(),
            },
            "target": torch.from_numpy(target.copy()).long(),
            "meta": {
                "sample_index": index,
                "tile_id": self.ids[tile_index],
            },
        }

    def _load_rgb_tile(self, tile_index: int) -> np.ndarray:
        if tile_index not in self.rgb_cache:
            rgb = np.asarray(imageio.imread(self.rgb_files[tile_index]), dtype=np.float32)
            required_channels = len(self.preset.rgb_channels)
            if rgb.ndim != 3 or rgb.shape[2] < required_channels:
                raise ValueError(
                    f"Expected RGB tile with at least {required_channels} channels, got {tuple(rgb.shape)} "
                    f"from {self.rgb_files[tile_index]}"
                )
            rgb = (rgb[:, :, self.preset.rgb_channels] / 255.0).transpose(2, 0, 1)
            if self.cache:
                self.rgb_cache[tile_index] = rgb
        if self.cache:
            rgb = self.rgb_cache[tile_index]
        else:
            rgb = (
                np.asarray(imageio.imread(self.rgb_files[tile_index]), dtype=np.float32)[:, :, self.preset.rgb_channels]
                / 255.0
            ).transpose(2, 0, 1)
        return rgb

    def _load_dsm_tile(self, tile_index: int) -> np.ndarray:
        if tile_index not in self.dsm_cache:
            dsm = DataUtils.normalize_dsm(imageio.imread(self.dsm_files[tile_index]))
            if self.cache:
                self.dsm_cache[tile_index] = dsm
        if self.cache:
            dsm = self.dsm_cache[tile_index]
        else:
            dsm = DataUtils.normalize_dsm(imageio.imread(self.dsm_files[tile_index]))
        return dsm

    def _load_eval_target_tile(self, tile_index: int) -> np.ndarray:
        return self._load_label_tile(
            tile_index=tile_index,
            files=self.eval_label_files,
            cache_store=self.eval_label_cache,
        )

    def _load_label_tile(
        self,
        tile_index: int,
        files: list[Path],
        cache_store: dict[int, np.ndarray],
    ) -> np.ndarray:
        if tile_index not in cache_store:
            target = DataUtils.convert_from_color(
                np.asarray(imageio.imread(files[tile_index])),
                invert_palette=self.preset.invert_palette,
            )
            if self.cache:
                cache_store[tile_index] = target
        if self.cache:
            target = cache_store[tile_index]
        else:
            target = DataUtils.convert_from_color(
                np.asarray(imageio.imread(files[tile_index])),
                invert_palette=self.preset.invert_palette,
            )
        return target

    def _resolve_tile_index(self, index: int) -> int:
        if self.split == "train":
            return random.randrange(len(self.ids))
        return index % len(self.ids)

    def _crop_random_patch(
        self, rgb: np.ndarray, dsm: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        patch_h, patch_w = self.patch_size
        height, width = rgb.shape[-2:]

        if self.split == "train":
            x1 = random.randint(0, height - patch_h - 1)
            y1 = random.randint(0, width - patch_w - 1)
        else:
            x1 = (height - patch_h) // 2
            y1 = (width - patch_w) // 2

        x2 = x1 + patch_h
        y2 = y1 + patch_w
        rgb_patch = rgb[:, x1:x2, y1:y2]
        dsm_patch = dsm[x1:x2, y1:y2]
        target_patch = target[x1:x2, y1:y2]
        return rgb_patch, dsm_patch, target_patch


def build_isprs_dataset(name: str, **kwargs: object) -> ISPRSDataset:
    dataset_name = str(name).strip().lower()
    if dataset_name == "vaihingen":
        return ISPRSDataset(preset=VAIHINGEN_PRESET, **kwargs)
    if dataset_name == "potsdam":
        return ISPRSDataset(preset=POTSDAM_PRESET, **kwargs)
    raise KeyError(f"Unsupported ISPRS dataset: {name!r}")
