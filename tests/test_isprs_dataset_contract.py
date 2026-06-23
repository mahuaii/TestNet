from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import numpy as np
from PIL import Image
import torch

from data import build_isprs_dataset
from data.isprs_dataset import ISPRSDataset, POTSDAM_PRESET, VAIHINGEN_PRESET
from engine import SlidingWindowInferencer
from multimodal_helpers import _WholeTileDataset
from utils import DataUtils


class ISPRSDatasetContractTest(unittest.TestCase):
    def _write_vaihingen_sample(self, root: Path, tile_id: str = "1") -> None:
        (root / "rgb").mkdir(exist_ok=True)
        (root / "dsm").mkdir(exist_ok=True)
        (root / "labels").mkdir(exist_ok=True)
        (root / "labels_eroded").mkdir(exist_ok=True)

        rgb = torch.zeros(32, 32, 3, dtype=torch.uint8).numpy()
        rgb[:, :, 0] = 255
        dsm = (torch.arange(32 * 32, dtype=torch.int16).reshape(32, 32)).numpy()
        label = torch.zeros(32, 32, 3, dtype=torch.uint8).numpy()
        label[:, :16] = torch.tensor([255, 255, 255], dtype=torch.uint8).numpy()
        label[:, 16:] = torch.tensor([0, 0, 255], dtype=torch.uint8).numpy()
        eroded_label = torch.zeros(32, 32, 3, dtype=torch.uint8).numpy()
        eroded_label[:, :] = torch.tensor([0, 0, 255], dtype=torch.uint8).numpy()

        Image.fromarray(rgb).save(root / "rgb" / f"top_mosaic_09cm_area{tile_id}.tif")
        Image.fromarray(dsm).save(root / "dsm" / f"dsm_09cm_matching_area{tile_id}.tif")
        Image.fromarray(label).save(root / "labels" / f"top_mosaic_09cm_area{tile_id}.tif")
        Image.fromarray(eroded_label).save(root / "labels_eroded" / f"top_mosaic_09cm_area{tile_id}_noBoundary.tif")

    def _write_potsdam_sample(self, root: Path, tile_id: str = "6_10") -> None:
        (root / "rgbir").mkdir()
        (root / "dsm").mkdir()
        (root / "labels").mkdir()
        (root / "labels_eroded").mkdir()

        rgbir = torch.zeros(32, 32, 4, dtype=torch.uint8).numpy()
        rgbir[:, :, 0] = 255
        rgbir[:, :, 1] = 64
        rgbir[:, :, 2] = 32
        rgbir[:, :, 3] = 200
        dsm = (torch.arange(32 * 32, dtype=torch.uint8).reshape(32, 32)).numpy()
        label = torch.zeros(32, 32, 3, dtype=torch.uint8).numpy()
        label[:, :16] = torch.tensor([255, 255, 255], dtype=torch.uint8).numpy()
        label[:, 16:] = torch.tensor([0, 0, 255], dtype=torch.uint8).numpy()
        eroded_label = label.copy()

        Image.fromarray(rgbir).save(root / "rgbir" / f"top_potsdam_{tile_id}_RGBIR.tif")
        Image.fromarray(dsm).save(root / "dsm" / f"dsm_potsdam_{tile_id}_normalized_lastools.jpg")
        Image.fromarray(label).save(root / "labels" / f"top_potsdam_{tile_id}_label.tif")
        Image.fromarray(eroded_label).save(root / "labels_eroded" / f"top_potsdam_{tile_id}_label_noBoundary.tif")

    def test_vaihingen_dataset_emits_real_training_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)

            random.seed(0)
            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                ids=["1"],
                dsm_preprocessing=None,
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=False,
                split="val",
            )

            sample = dataset[0]

            self.assertIn("inputs", sample)
            self.assertEqual(sample["inputs"]["rgb"].shape, (3, 16, 16))
            self.assertEqual(sample["inputs"]["dsm"].shape, (16, 16))
            self.assertEqual(sample["target"].shape, (16, 16))
            self.assertTrue(set(torch.unique(sample["target"]).tolist()).issubset({0, 1}))
            raw_dsm = torch.arange(32 * 32, dtype=torch.int16).reshape(32, 32).numpy()
            expected_dsm = DataUtils.normalize_dsm(raw_dsm)[8:24, 8:24]
            np.testing.assert_allclose(sample["inputs"]["dsm"].numpy(), expected_dsm, rtol=1e-6, atol=1e-6)

    def test_vaihingen_dataset_augments_cropped_patch_not_full_tile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)

            random.seed(0)
            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                ids=["1"],
                dsm_preprocessing=None,
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=True,
                split="train",
            )

            sample = dataset[0]

            self.assertEqual(sample["inputs"]["rgb"].shape, (3, 16, 16))
            self.assertEqual(sample["inputs"]["dsm"].shape, (16, 16))
            self.assertEqual(sample["target"].shape, (16, 16))

    def test_training_dataset_uses_configured_tile_sampling_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root, tile_id="1")
            self._write_vaihingen_sample(root, tile_id="3")
            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                ids=["1", "3"],
                dsm_preprocessing=None,
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=False,
                split="train",
                tile_sampling_weights=[1.0, 4.0],
            )

            with patch("data.isprs_dataset.random.choices", return_value=[1]) as choices:
                tile_index = dataset._resolve_tile_index(0)

            self.assertEqual(tile_index, 1)
            choices.assert_called_once_with(range(2), weights=(1.0, 4.0), k=1)

    def test_training_dataset_without_weights_keeps_uniform_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)
            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                ids=["1"],
                dsm_preprocessing=None,
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=False,
                split="train",
            )

            with patch("data.isprs_dataset.random.randrange", return_value=0) as randrange:
                tile_index = dataset._resolve_tile_index(0)

            self.assertEqual(tile_index, 0)
            randrange.assert_called_once_with(1)

    def test_disabled_patch_sampling_keeps_random_crop_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)
            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                ids=["1"],
                dsm_preprocessing=None,
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=False,
                split="train",
                patch_sampling={"enabled": False},
            )
            rgb = np.zeros((3, 32, 32), dtype=np.float32)
            dsm = np.zeros((32, 32), dtype=np.float32)
            target = np.zeros((32, 32), dtype=np.int64)

            with patch("data.isprs_dataset.random.random") as random_draw:
                with patch("data.isprs_dataset.random.randint", side_effect=[3, 5]):
                    _, _, target_patch = dataset._crop_random_patch(rgb, dsm, target)

            random_draw.assert_not_called()
            self.assertEqual(target_patch.shape, (16, 16))

    def test_enabled_patch_sampling_can_prefer_car_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)
            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                ids=["1"],
                dsm_preprocessing=None,
                patch_size=(4, 4),
                samples_per_epoch=1,
                cache=True,
                augmentation=False,
                split="train",
                patch_sampling={
                    "enabled": True,
                    "uniform_probability": 0.0,
                    "car_probability": 1.0,
                    "boundary_probability": 0.0,
                    "num_candidates": 2,
                    "min_car_pixels": 1,
                },
            )
            rgb = np.zeros((3, 8, 8), dtype=np.float32)
            dsm = np.zeros((8, 8), dtype=np.float32)
            target = np.zeros((8, 8), dtype=np.int64)
            target[4:8, 4:8] = 4

            with patch("data.isprs_dataset.random.random", return_value=0.5):
                with patch("data.isprs_dataset.random.randint", side_effect=[0, 0, 4, 4]):
                    with patch(
                        "data.isprs_dataset.random.choices",
                        return_value=[(4, 4)],
                    ) as choices:
                        _, _, target_patch = dataset._crop_random_patch(rgb, dsm, target)

            self.assertTrue(np.all(target_patch == 4))
            self.assertEqual(choices.call_args.kwargs["weights"], [0, 16])

    def test_enabled_patch_sampling_can_prefer_boundary_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)
            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                ids=["1"],
                dsm_preprocessing=None,
                patch_size=(4, 4),
                samples_per_epoch=1,
                cache=True,
                augmentation=False,
                split="train",
                patch_sampling={
                    "enabled": True,
                    "uniform_probability": 0.0,
                    "car_probability": 0.0,
                    "boundary_probability": 1.0,
                    "num_candidates": 2,
                },
            )
            target = np.zeros((8, 8), dtype=np.int64)
            target[4:8, 6:8] = 1

            with patch("data.isprs_dataset.random.random", return_value=0.5):
                with patch("data.isprs_dataset.random.randint", side_effect=[0, 0, 4, 4]):
                    with patch(
                        "data.isprs_dataset.random.choices",
                        return_value=[(4, 4)],
                    ) as choices:
                        coordinates = dataset._sample_prioritized_crop(target, 8, 8)

            self.assertEqual(coordinates, (4, 4))
            self.assertGreater(choices.call_args.kwargs["weights"][1], 0)

    def test_dataset_rejects_invalid_tile_sampling_weights(self) -> None:
        invalid_cases = [
            ([1.0, 2.0], ValueError, "length must match"),
            ([-1.0], ValueError, "non-negative"),
            ([float("inf")], ValueError, "finite"),
            ([float("nan")], ValueError, "finite"),
            ([0.0], ValueError, "at least one positive"),
            (["1.0"], TypeError, "must be a number"),
        ]

        for weights, error_type, message in invalid_cases:
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(error_type, message):
                    ISPRSDataset(
                        preset=VAIHINGEN_PRESET,
                        root_dir="/unused",
                        ids=["1"],
                        dsm_preprocessing=None,
                        tile_sampling_weights=weights,
                    )

    def test_vaihingen_dataset_get_tile_returns_uncropped_full_tile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)

            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                ids=["1"],
                dsm_preprocessing=None,
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=True,
                split="val",
            )

            sample = dataset.get_tile(0)

            self.assertEqual(sample["inputs"]["rgb"].shape, (3, 32, 32))
            self.assertEqual(sample["inputs"]["dsm"].shape, (32, 32))
            self.assertEqual(sample["target"].shape, (32, 32))
            self.assertEqual(sample["meta"]["tile_id"], "1")
            self.assertEqual(set(torch.unique(sample["target"]).tolist()), {1})

    def test_vaihingen_dataset_applies_enabled_dsm_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)
            dsm_preprocessing = {
                "enabled": True,
                "type": "similarity_enhancement",
                "window_size": 3,
                "sigma": 0.2,
                "lambda_weight": 0.4,
            }
            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                ids=["1"],
                dsm_preprocessing=dsm_preprocessing,
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=False,
                split="val",
            )

            sample = dataset.get_tile(0)

            raw_dsm = torch.arange(32 * 32, dtype=torch.int16).reshape(32, 32).numpy()
            expected_dsm = DataUtils.enhance_dsm_similarity(
                raw_dsm,
                window_size=3,
                sigma=0.2,
                lambda_weight=0.4,
            )
            np.testing.assert_allclose(sample["inputs"]["dsm"].numpy(), expected_dsm, rtol=1e-6, atol=1e-6)

    def test_vaihingen_dataset_accepts_null_dsm_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)
            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                ids=["1"],
                dsm_preprocessing=None,
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=False,
                split="val",
            )

            sample = dataset.get_tile(0)

            raw_dsm = torch.arange(32 * 32, dtype=torch.int16).reshape(32, 32).numpy()
            expected_dsm = DataUtils.normalize_dsm(raw_dsm)
            np.testing.assert_allclose(sample["inputs"]["dsm"].numpy(), expected_dsm, rtol=1e-6, atol=1e-6)

    def test_vaihingen_dataset_rejects_unknown_dsm_preprocessing_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)

            with self.assertRaisesRegex(ValueError, "Unsupported DSM preprocessing type"):
                ISPRSDataset(
                    preset=VAIHINGEN_PRESET,
                    root_dir=str(root),
                    ids=["1"],
                    dsm_preprocessing={
                        "enabled": True,
                        "type": "unknown",
                        "window_size": 3,
                        "sigma": 0.2,
                        "lambda_weight": 0.4,
                    },
                    patch_size=(16, 16),
                    samples_per_epoch=1,
                    cache=True,
                    augmentation=False,
                    split="val",
                )

    def test_vaihingen_dataset_accepts_disabled_dsm_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)
            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                ids=["1"],
                dsm_preprocessing={
                    "enabled": False,
                    "type": "similarity_enhancement",
                    "window_size": 3,
                    "sigma": 0.2,
                    "lambda_weight": 0.4,
                },
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=False,
                split="val",
            )

            sample = dataset.get_tile(0)

            raw_dsm = torch.arange(32 * 32, dtype=torch.int16).reshape(32, 32).numpy()
            expected_dsm = DataUtils.normalize_dsm(raw_dsm)
            np.testing.assert_allclose(sample["inputs"]["dsm"].numpy(), expected_dsm, rtol=1e-6, atol=1e-6)

    def test_whole_tile_sliding_window_matches_original_mfnet_path(self) -> None:
        rgb = torch.arange(3 * 5 * 5, dtype=torch.float32).reshape(3, 5, 5)
        dsm = torch.arange(5 * 5, dtype=torch.float32).reshape(5, 5)
        target = torch.zeros(5, 5, dtype=torch.long)
        dataset = _WholeTileDataset(
            {
                "inputs": {"rgb": rgb, "dsm": dsm},
                "target": target,
                "meta": {"tile_id": "1"},
            }
        )
        modes: list[str] = []

        def model(rgb: torch.Tensor, dsm: torch.Tensor, mode: str = "Train") -> torch.Tensor:
            del dsm
            modes.append(str(mode))
            return torch.stack([torch.zeros_like(rgb[:, 0]), torch.ones_like(rgb[:, 0])], dim=1)

        outputs = SlidingWindowInferencer().run(
            model=model,
            dataset=dataset,
            device=torch.device("cpu"),
            stride=3,
            batch_size=2,
            window_size=(3, 3),
            num_classes=2,
            input_modals=("rgb", "dsm"),
            model_kwargs={"mode": "Test"},
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["pred"].shape, (5, 5))
        self.assertTrue(torch.equal(outputs[0]["pred"], torch.ones(5, 5, dtype=torch.long)))
        self.assertIs(outputs[0]["target"], target)
        self.assertEqual(outputs[0]["meta"], {"tile_id": "1"})
        self.assertEqual(dataset.requested_indices, [0])
        self.assertTrue(modes)
        self.assertTrue(all(mode == "Test" for mode in modes))

    def test_whole_tile_sliding_window_uses_explicit_rgb_dsm_order(self) -> None:
        rgb = torch.arange(3 * 5 * 5, dtype=torch.float32).reshape(3, 5, 5)
        dsm = torch.arange(5 * 5, dtype=torch.float32).reshape(5, 5)
        target = torch.zeros(5, 5, dtype=torch.long)
        dataset = _WholeTileDataset(
            {
                "inputs": {"dsm": dsm, "rgb": rgb},
                "target": target,
                "meta": {"tile_id": "1"},
            }
        )
        modes: list[str] = []
        call_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def model(rgb: torch.Tensor, dsm: torch.Tensor, mode: str = "Train") -> torch.Tensor:
            modes.append(str(mode))
            call_shapes.append((tuple(rgb.shape), tuple(dsm.shape)))
            return torch.stack([torch.zeros_like(rgb[:, 0]), torch.ones_like(rgb[:, 0])], dim=1)

        SlidingWindowInferencer().run(
            model=model,
            dataset=dataset,
            device=torch.device("cpu"),
            stride=3,
            batch_size=2,
            window_size=(3, 3),
            num_classes=2,
            input_modals=("rgb", "dsm"),
            model_kwargs={"mode": "Test"},
        )

        self.assertTrue(modes)
        self.assertTrue(all(mode == "Test" for mode in modes))
        self.assertEqual(call_shapes[0], ((2, 3, 3, 3), (2, 3, 3)))

    def test_whole_tile_sliding_window_accepts_generic_multimodal_inputs(self) -> None:
        rgb = torch.arange(3 * 5 * 5, dtype=torch.float32).reshape(3, 5, 5)
        dsm = torch.arange(5 * 5, dtype=torch.float32).reshape(5, 5)
        target = torch.zeros(5, 5, dtype=torch.long)
        dataset = _WholeTileDataset(
            {
                "inputs": {"rgb": rgb, "dsm": dsm},
                "target": target,
                "meta": {"tile_id": "1"},
            }
        )
        modes: list[str] = []

        def model(rgb: torch.Tensor, dsm: torch.Tensor, mode: str = "Train") -> torch.Tensor:
            del dsm
            modes.append(str(mode))
            return torch.stack([torch.zeros_like(rgb[:, 0]), torch.ones_like(rgb[:, 0])], dim=1)

        outputs = SlidingWindowInferencer().run(
            model=model,
            dataset=dataset,
            device=torch.device("cpu"),
            stride=3,
            batch_size=2,
            window_size=(3, 3),
            num_classes=2,
            input_modals=("rgb", "dsm"),
            model_kwargs={"mode": "Test"},
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["pred"].shape, (5, 5))
        self.assertTrue(torch.equal(outputs[0]["pred"], torch.ones(5, 5, dtype=torch.long)))
        self.assertIs(outputs[0]["target"], target)
        self.assertEqual(outputs[0]["meta"], {"tile_id": "1"})
        self.assertEqual(dataset.requested_indices, [0])
        self.assertTrue(modes)
        self.assertTrue(all(mode == "Test" for mode in modes))

    def test_whole_tile_sliding_window_accepts_single_input_key(self) -> None:
        image = torch.arange(3 * 5 * 5, dtype=torch.float32).reshape(3, 5, 5)
        target = torch.zeros(5, 5, dtype=torch.long)
        dataset = _WholeTileDataset(
            {
                "inputs": {"image": image},
                "target": target,
                "meta": {"tile_id": "1"},
            }
        )
        batch_shapes: list[tuple[int, ...]] = []

        def model(image: torch.Tensor) -> torch.Tensor:
            batch_shapes.append(tuple(image.shape))
            return torch.stack([torch.zeros_like(image[:, 0]), torch.ones_like(image[:, 0])], dim=1)

        outputs = SlidingWindowInferencer().run(
            model=model,
            dataset=dataset,
            device=torch.device("cpu"),
            stride=3,
            batch_size=2,
            window_size=(3, 3),
            num_classes=2,
            input_modals=("image",),
        )

        self.assertEqual(len(outputs), 1)
        self.assertTrue(torch.equal(outputs[0]["pred"], torch.ones(5, 5, dtype=torch.long)))
        self.assertIs(outputs[0]["target"], target)
        self.assertEqual(outputs[0]["meta"], {"tile_id": "1"})
        self.assertEqual(dataset.requested_indices, [0])
        self.assertEqual(batch_shapes[0], (2, 3, 3, 3))

    def test_potsdam_dataset_emits_same_training_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_potsdam_sample(root)

            random.seed(0)
            dataset = ISPRSDataset(
                preset=POTSDAM_PRESET,
                root_dir=str(root),
                ids=["6_10"],
                dsm_preprocessing=None,
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=False,
                split="val",
            )

            sample = dataset[0]

            self.assertIn("inputs", sample)
            self.assertEqual(sample["inputs"]["rgb"].shape, (3, 16, 16))
            self.assertEqual(sample["inputs"]["dsm"].shape, (16, 16))
            self.assertEqual(sample["target"].shape, (16, 16))
            self.assertTrue(set(torch.unique(sample["target"]).tolist()).issubset({0, 1}))

    def test_training_crop_bounds_match_original_random_pos(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_vaihingen_sample(root)
            dataset = ISPRSDataset(
                preset=VAIHINGEN_PRESET,
                root_dir=str(root),
                ids=["1"],
                dsm_preprocessing=None,
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=False,
                split="train",
            )

            with patch("data.isprs_dataset.random.randint", side_effect=[3, 5]) as randint:
                sample = dataset[0]

            self.assertEqual(randint.call_args_list, [call(0, 15), call(0, 15)])
            self.assertEqual(sample["inputs"]["rgb"].shape, (3, 16, 16))
            self.assertEqual(sample["inputs"]["dsm"].shape, (16, 16))

    def test_build_isprs_dataset_dispatches_to_potsdam(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_potsdam_sample(root)

            dataset = build_isprs_dataset(
                "potsdam",
                root_dir=str(root),
                ids=["6_10"],
                dsm_preprocessing=None,
                patch_size=(16, 16),
                samples_per_epoch=1,
                cache=True,
                augmentation=False,
                split="val",
            )

            self.assertIsInstance(dataset, ISPRSDataset)
            self.assertEqual(dataset.preset.name, "potsdam")
            sample = dataset[0]
            self.assertEqual(sample["inputs"]["rgb"].shape, (3, 16, 16))
