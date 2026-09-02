from __future__ import annotations

import itertools
from typing import Any

import torch
from tqdm import tqdm

from .sliding_window_inferencer import SlidingWindowInferencer


class SlidingWindowInferencer3090(SlidingWindowInferencer):
    """Sliding-window inference with GPU-side logit accumulation."""

    @torch.inference_mode()
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
        outputs: list[dict[str, Any]] = []

        tile_iterable = tqdm(
            range(len(dataset.ids)),
            total=len(dataset.ids),
            desc="Validation tiles",
            leave=False,
        )

        for tile_index in tile_iterable:
            tile = dataset.get_tile(tile_index)
            inputs = tile["inputs"]
            target = tile["target"]
            input_tensors = [(modal, inputs[modal]) for modal in input_modals]

            height, width = input_tensors[0][1].shape[-2:]
            accumulated_logits = torch.zeros(
                (height, width, num_classes),
                device=device,
                dtype=torch.float32,
            )

            coordinates = self._get_sliding_window_coords(
                (height, width),
                step=stride,
                window_size=window_size,
            )
            coord_iterable = tqdm(
                self._grouper(batch_size, coordinates),
                total=self._count_sliding_window_batches(
                    (height, width),
                    step=stride,
                    window_size=window_size,
                    batch_size=batch_size,
                ),
                desc=f"Tile {tile_index + 1}",
                leave=False,
            )

            for coords in coord_iterable:
                batch_inputs = [
                    self._crop_batch(input_tensor, coords, device)
                    for _, input_tensor in input_tensors
                ]
                model_output = model(*batch_inputs, **model_kwargs)
                logits = self._extract_logits(model_output).float()
                self._accumulate_logits(accumulated_logits, logits, coords)

            pred = accumulated_logits.argmax(dim=-1).cpu().long()
            outputs.append(
                {
                    "pred": pred,
                    "target": target,
                    "meta": tile.get("meta", {}),
                }
            )

        return outputs

    @staticmethod
    def _crop_batch(
        tensor: torch.Tensor,
        coords: Any,
        device: torch.device,
    ) -> torch.Tensor:
        patches = torch.stack(
            [tensor[..., x : x + w, y : y + h] for x, y, w, h in coords],
            dim=0,
        )
        if device.type == "cuda":
            patches = patches.pin_memory()
        return patches.to(device, non_blocking=True)

    @staticmethod
    def _accumulate_logits(
        accumulated_logits: torch.Tensor,
        logits: torch.Tensor,
        coords: Any,
    ) -> None:
        for logit, (x, y, w, h) in zip(logits, coords):
            accumulated_logits[x : x + w, y : y + h] += logit.permute(1, 2, 0)

    @staticmethod
    def _extract_logits(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict) and "logits" in output:
            logits = output["logits"]
            if isinstance(logits, torch.Tensor):
                return logits
        raise TypeError("3090 sliding-window inference expects a tensor or a mapping with tensor 'logits'.")

    @staticmethod
    def _grouper(n: int, iterable: Any) -> Any:
        iterator = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(iterator, n))
            if not chunk:
                return
            yield chunk


__all__ = ["SlidingWindowInferencer3090"]
