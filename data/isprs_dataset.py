from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
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
        dsm_preprocessing: Mapping[str, object],
        patch_size: Sequence[int] = (256, 256),
        samples_per_epoch: int | None = None,
        cache: bool = True,
        augmentation: bool = True,
        split: str = "train",
        tile_sampling_weights: Sequence[float] | None = None,
        patch_sampling: Mapping[str, object] | None = None,
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
        self.dsm_preprocessing = self._parse_dsm_preprocessing(dsm_preprocessing)
        self.tile_sampling_weights = self._parse_tile_sampling_weights(tile_sampling_weights)
        self.patch_sampling = self._parse_patch_sampling(patch_sampling)

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
            dsm = self._preprocess_dsm(imageio.imread(self.dsm_files[tile_index]))
            if self.cache:
                self.dsm_cache[tile_index] = dsm
        if self.cache:
            dsm = self.dsm_cache[tile_index]
        else:
            dsm = self._preprocess_dsm(imageio.imread(self.dsm_files[tile_index]))
        return dsm

    @staticmethod
    def _parse_dsm_preprocessing(dsm_preprocessing: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(dsm_preprocessing, Mapping):
            raise TypeError(
                "Expected dsm_preprocessing to be a mapping, "
                f"got {type(dsm_preprocessing).__name__}."
            )

        enabled = dsm_preprocessing["enabled"]
        if not isinstance(enabled, bool):
            raise TypeError(
                "Expected dsm_preprocessing.enabled to be a bool, "
                f"got {type(enabled).__name__}."
            )

        preprocessing_type = dsm_preprocessing["type"]
        if preprocessing_type != "similarity_enhancement":
            raise ValueError(f"Unsupported DSM preprocessing type: {preprocessing_type!r}")

        return dict(dsm_preprocessing)

    def _preprocess_dsm(self, dsm: np.ndarray) -> np.ndarray:
        if not self.dsm_preprocessing["enabled"]:
            return DataUtils.normalize_dsm(dsm)

        return DataUtils.enhance_dsm_similarity(
            dsm,
            window_size=self.dsm_preprocessing["window_size"],
            sigma=self.dsm_preprocessing["sigma"],
            lambda_weight=self.dsm_preprocessing["lambda_weight"],
        )

    def _parse_tile_sampling_weights(
        self,
        tile_sampling_weights: Sequence[float] | None,
    ) -> tuple[float, ...] | None:
        if tile_sampling_weights is None:
            return None
        if isinstance(tile_sampling_weights, (str, bytes)) or not isinstance(
            tile_sampling_weights, Sequence
        ):
            raise TypeError("tile_sampling_weights must be a sequence of numbers")
        if len(tile_sampling_weights) != len(self.ids):
            raise ValueError(
                "tile_sampling_weights length must match ids length: "
                f"got {len(tile_sampling_weights)} weights for {len(self.ids)} tile ids"
            )

        weights: list[float] = []
        for index, weight in enumerate(tile_sampling_weights):
            if isinstance(weight, bool) or not isinstance(weight, Real):
                raise TypeError(
                    f"tile_sampling_weights[{index}] must be a number, "
                    f"got {type(weight).__name__}"
                )
            numeric_weight = float(weight)
            if not math.isfinite(numeric_weight):
                raise ValueError(f"tile_sampling_weights[{index}] must be finite")
            if numeric_weight < 0:
                raise ValueError(f"tile_sampling_weights[{index}] must be non-negative")
            weights.append(numeric_weight)

        if not any(weight > 0 for weight in weights):
            raise ValueError("tile_sampling_weights must contain at least one positive weight")
        return tuple(weights)

    @staticmethod
    def _parse_patch_sampling(patch_sampling: Mapping[str, object] | None) -> dict[str, object]:
        defaults: dict[str, object] = {
            "enabled": False,
            "uniform_probability": 0.5,
            "car_probability": 0.3,
            "boundary_probability": 0.2,
            "num_candidates": 10,
            "car_class_id": 4,
            "min_car_pixels": 16,
            "ignore_index": 255,
        }
        if patch_sampling is None:
            return defaults
        if not isinstance(patch_sampling, Mapping):
            raise TypeError("patch_sampling must be a mapping")

        config = {**defaults, **patch_sampling}
        if not isinstance(config["enabled"], bool):
            raise TypeError("patch_sampling.enabled must be a bool")

        probabilities = []
        for key in ("uniform_probability", "car_probability", "boundary_probability"):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"patch_sampling.{key} must be a number")
            probability = float(value)
            if not math.isfinite(probability) or probability < 0:
                raise ValueError(f"patch_sampling.{key} must be finite and non-negative")
            config[key] = probability
            probabilities.append(probability)
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("patch_sampling probabilities must sum to 1")

        for key in ("num_candidates", "car_class_id", "min_car_pixels", "ignore_index"):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"patch_sampling.{key} must be an integer")
        if config["num_candidates"] <= 0:
            raise ValueError("patch_sampling.num_candidates must be positive")
        if config["car_class_id"] < 0 or config["min_car_pixels"] < 0:
            raise ValueError("patch_sampling car settings must be non-negative")
        return config

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
            if self.tile_sampling_weights is not None:
                return random.choices(
                    range(len(self.ids)),
                    weights=self.tile_sampling_weights,
                    k=1,
                )[0]
            return random.randrange(len(self.ids))
        return index % len(self.ids)

    def _crop_random_patch(
        self, rgb: np.ndarray, dsm: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        patch_h, patch_w = self.patch_size
        height, width = rgb.shape[-2:]

        if self.split == "train" and self.patch_sampling["enabled"]:
            x1, y1 = self._sample_prioritized_crop(target, height, width)
        elif self.split == "train":
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

    def _sample_prioritized_crop(
        self,
        target: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[int, int]:
        patch_h, patch_w = self.patch_size
        max_x = height - patch_h
        max_y = width - patch_w
        draw = random.random()
        uniform_threshold = float(self.patch_sampling["uniform_probability"])
        car_threshold = uniform_threshold + float(self.patch_sampling["car_probability"])

        if draw < uniform_threshold:
            return random.randint(0, max_x), random.randint(0, max_y)

        candidates = [
            (random.randint(0, max_x), random.randint(0, max_y))
            for _ in range(int(self.patch_sampling["num_candidates"]))
        ]
        if draw < car_threshold:
            car_class_id = int(self.patch_sampling["car_class_id"])
            min_car_pixels = int(self.patch_sampling["min_car_pixels"])
            scores = [
                int(np.count_nonzero(target[x : x + patch_h, y : y + patch_w] == car_class_id))
                for x, y in candidates
            ]
            scores = [score if score >= min_car_pixels else 0 for score in scores]
        else:
            ignore_index = int(self.patch_sampling["ignore_index"])
            boundary = np.zeros_like(target, dtype=bool)
            vertical = (
                (target[1:, :] != target[:-1, :])
                & (target[1:, :] != ignore_index)
                & (target[:-1, :] != ignore_index)
            )
            horizontal = (
                (target[:, 1:] != target[:, :-1])
                & (target[:, 1:] != ignore_index)
                & (target[:, :-1] != ignore_index)
            )
            boundary[1:, :] |= vertical
            boundary[:-1, :] |= vertical
            boundary[:, 1:] |= horizontal
            boundary[:, :-1] |= horizontal
            scores = [
                int(np.count_nonzero(boundary[x : x + patch_h, y : y + patch_w]))
                for x, y in candidates
            ]

        if any(scores):
            return random.choices(candidates, weights=scores, k=1)[0]
        return random.choice(candidates)


def build_isprs_dataset(name: str, **kwargs: object) -> ISPRSDataset:
    dataset_name = str(name).strip().lower()
    if dataset_name == "vaihingen":
        return ISPRSDataset(preset=VAIHINGEN_PRESET, **kwargs)
    if dataset_name == "potsdam":
        return ISPRSDataset(preset=POTSDAM_PRESET, **kwargs)
    raise KeyError(f"Unsupported ISPRS dataset: {name!r}")
