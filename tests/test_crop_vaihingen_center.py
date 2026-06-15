from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from tools.crop_vaihingen_center import (
    build_crop_tasks,
    center_crop_origin,
    crop_vaihingen_center,
    discover_common_area_ids,
    output_name,
    parse_area_id,
)


_SOURCE_NAMES = {
    "rgb": "top_mosaic_09cm_area{area_id}.tif",
    "dsm": "dsm_09cm_matching_area{area_id}.tif",
    "labels": "top_mosaic_09cm_area{area_id}.tif",
    "labels_eroded": "top_mosaic_09cm_area{area_id}_noBoundary.tif",
}


class CropVaihingenCenterTest(unittest.TestCase):
    def test_parsing_naming_and_center_origin(self) -> None:
        self.assertEqual(parse_area_id("top_mosaic_09cm_area23_noBoundary.tif"), 23)
        self.assertEqual(output_name("dsm", 23), "dsm_09cm_matching_area123.tif")
        self.assertEqual(
            output_name("labels_eroded", 1),
            "top_mosaic_09cm_area101_noBoundary.tif",
        )
        self.assertEqual(center_crop_origin(7, 9, 4), (1, 2))

    def test_discovery_uses_four_modality_intersection_and_ignores_generated_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_directories(root)
            self._write_area(root, 1, size=6)
            self._write_area(root, 2, size=6)
            self._write_single(root, "labels_eroded", 9, np.zeros((6, 6, 3), dtype=np.uint8))
            self._write_area(root, 51, size=4)

            self.assertEqual(discover_common_area_ids(root), [1, 2])

    def test_build_tasks_rejects_mismatched_modality_dimensions_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_directories(root)
            self._write_area(root, 1, size=6)
            self._write_single(root, "dsm", 1, np.zeros((7, 6), dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "dimensions do not match"):
                build_crop_tasks(root, crop_size=4)
            self.assertFalse((root / "rgb" / "top_mosaic_09cm_area101.tif").exists())

    def test_build_tasks_rejects_source_smaller_than_crop_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_directories(root)
            self._write_area(root, 1, size=3)

            with self.assertRaisesRegex(ValueError, "smaller than crop size"):
                build_crop_tasks(root, crop_size=4)
            self.assertFalse((root / "rgb" / "top_mosaic_09cm_area101.tif").exists())

    def test_crop_preserves_values_shapes_dtypes_and_overwrites_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_directories(root)
            source_arrays = self._write_area(root, 1, size=6)

            first_tasks = crop_vaihingen_center(root, crop_size=4)
            self.assertEqual(len(first_tasks), 4)
            for modality, source in source_arrays.items():
                target = root / modality / output_name(modality, 1)
                actual = tifffile.imread(target)
                np.testing.assert_array_equal(actual, source[1:5, 1:5, ...])
                self.assertEqual(actual.dtype, source.dtype)

            replacement_arrays = self._write_area(root, 1, size=6, value_offset=50)
            second_tasks = crop_vaihingen_center(root, crop_size=4)
            self.assertEqual(len(second_tasks), 4)
            for modality, source in replacement_arrays.items():
                target = root / modality / output_name(modality, 1)
                actual = tifffile.imread(target)
                np.testing.assert_array_equal(actual, source[1:5, 1:5, ...])

    @staticmethod
    def _create_directories(root: Path) -> None:
        for modality in _SOURCE_NAMES:
            (root / modality).mkdir()

    def _write_area(
        self,
        root: Path,
        area_id: int,
        size: int,
        value_offset: int = 0,
    ) -> dict[str, np.ndarray]:
        grid = np.arange(size * size, dtype=np.uint8).reshape(size, size)
        arrays = {
            "rgb": np.stack([grid, grid + 1, grid + 2], axis=-1) + value_offset,
            "dsm": grid.astype(np.float32) + 0.25 + value_offset,
            "labels": np.stack([grid, grid, grid], axis=-1) + value_offset,
            "labels_eroded": np.stack([grid + 3, grid + 3, grid + 3], axis=-1)
            + value_offset,
        }
        for modality, image in arrays.items():
            self._write_single(root, modality, area_id, image)
        return arrays

    @staticmethod
    def _write_single(root: Path, modality: str, area_id: int, image: np.ndarray) -> None:
        path = root / modality / _SOURCE_NAMES[modality].format(area_id=area_id)
        tifffile.imwrite(path, image)


if __name__ == "__main__":
    unittest.main()
