from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


class SlidingWindowInferencer:
    """
    输入：
    - model
    - 支持 get_tile 的 dataset
    - device

    职责：
    - 承载整图滑窗推理流程
    - 支持单模态/多模态输入

    输出：
    - 整套 tile 的推理结果
    """

    @torch.no_grad()
    def run(
        self,
        model: Any,
        dataset: Any,
        device: torch.device,
        stride: int,
        batch_size: int,
        window_size: tuple[int, int],
        num_classes: int,
        input_modals: tuple[str, ...],
        model_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        model_kwargs = model_kwargs or {}
        outputs = []

        tile_iterable = tqdm(
            range(len(dataset.tile_names)),
            total=len(dataset.tile_names),
            desc="Validation tiles",
            leave=False,
        )

        for tile_index in tile_iterable:
            tile = dataset.get_tile(tile_index)
            inputs = tile["inputs"]
            target = tile["target"]
            input_tensors = [(modal, inputs[modal]) for modal in input_modals]

            height, width = input_tensors[0][1].shape[-2:]
            pred = np.zeros((height, width, num_classes))

            coord_iterable = self._grouper(
                batch_size,
                self._get_sliding_window_coords((height, width), step=stride, window_size=window_size),
            )
            coord_iterable = tqdm(
                coord_iterable,
                total=self._count_sliding_window_batches(
                    (height, width), step=stride, window_size=window_size, batch_size=batch_size
                ),
                desc=f"Tile {tile_index + 1}",
                leave=False,
            )

            for coords in coord_iterable:
                batch_inputs = [self._crop_batch(input_tensor, coords, device) for _, input_tensor in input_tensors]
                logits = model(*batch_inputs, **model_kwargs)
                logits = logits.data.cpu().numpy()
                self._accumulate_logits(pred, logits, coords)
                del logits

            pred = np.argmax(pred, axis=-1)
            outputs.append(
                {
                    "pred": torch.from_numpy(pred).long(),
                    "target": target,
                    "meta": tile.get("meta", {}),
                }
            )
        return outputs

    @staticmethod
    def _get_tile_by_index(dataset: Any, tile_index: int) -> dict[str, Any]:
        return dataset.get_tile(tile_index)

    @staticmethod
    def _crop_batch(tensor: torch.Tensor, coords: Any, device: torch.device) -> torch.Tensor:
        patches = [np.copy(tensor[..., x : x + w, y : y + h].numpy()) for x, y, w, h in coords]
        patches = np.asarray(patches)
        return torch.from_numpy(patches).to(device)

    @staticmethod
    def _accumulate_logits(pred: np.ndarray, logits: np.ndarray, coords: Any):
        for logit, (x, y, w, h) in zip(logits, coords):
            logit = logit.transpose((1, 2, 0))
            pred[x : x + w, y : y + h] += logit

    @staticmethod
    def _get_sliding_window_coords(
        image_shape: tuple[int, int],
        step: int,
        window_size: tuple[int, int],
    ) -> Any:
        for x in range(0, image_shape[0], step):
            if x + window_size[0] > image_shape[0]:
                x = image_shape[0] - window_size[0]
            for y in range(0, image_shape[1], step):
                if y + window_size[1] > image_shape[1]:
                    y = image_shape[1] - window_size[1]
                yield x, y, window_size[0], window_size[1]

    @classmethod
    def _count_sliding_window_batches(
        cls,
        image_shape: tuple[int, int],
        step: int,
        window_size: tuple[int, int],
        batch_size: int,
    ) -> int:
        window_count = sum(1 for _ in cls._get_sliding_window_coords(image_shape, step=step, window_size=window_size))
        return int(np.ceil(window_count / float(batch_size)))

    @staticmethod
    def _grouper(n: int, iterable: Any) -> Any:
        iterator = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(iterator, n))
            if not chunk:
                return
            yield chunk
