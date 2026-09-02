from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


class DeviceMetricAccumulator:
    """Accumulate scalar training metrics on the active device.

    Values are transferred to the host only when ``snapshot`` is requested.
    Accuracy is represented by its numerator and denominator so it does not
    require a host synchronization for every batch.
    """

    _ACCURACY_CORRECT = "accuracy_correct"
    _ACCURACY_TOTAL = "accuracy_total"

    def __init__(self) -> None:
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}
        self._accuracy_correct: torch.Tensor | None = None
        self._accuracy_total: torch.Tensor | None = None
        self._device: torch.device | None = None

    def update(self, metrics: Mapping[str, Any]) -> None:
        for value in metrics.values():
            if isinstance(value, torch.Tensor):
                value_device = value.device
                if self._device is None:
                    self._device = value_device
                elif value_device != self._device:
                    raise RuntimeError(
                        "DeviceMetricAccumulator received metrics from multiple devices"
                    )
                break

        for name, value in metrics.items():
            if name == self._ACCURACY_CORRECT:
                self._accuracy_correct = self._add(
                    self._accuracy_correct,
                    value,
                )
                continue
            if name == self._ACCURACY_TOTAL:
                self._accuracy_total = self._add(
                    self._accuracy_total,
                    value,
                )
                continue

            scalar = self._as_scalar(value)
            if name in self._sums:
                self._sums[name] = self._sums[name] + scalar
            else:
                self._sums[name] = scalar
            self._counts[name] = self._counts.get(name, 0) + 1

    def snapshot(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, value in self._sums.items():
            result[name] = float((value / self._counts[name]).detach().cpu().item())

        if self._accuracy_correct is not None and self._accuracy_total is not None:
            accuracy = self._accuracy_correct / self._accuracy_total.clamp_min(1.0)
            result["accuracy"] = float((100.0 * accuracy).detach().cpu().item())
        return result

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()
        self._accuracy_correct = None
        self._accuracy_total = None
        self._device = None

    def _add(self, current: torch.Tensor | None, value: Any) -> torch.Tensor:
        scalar = self._as_scalar(value)
        return scalar if current is None else current + scalar

    def _as_scalar(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("DeviceMetricAccumulator only accepts scalar tensors")
            scalar = value.detach().float()
            if self._device is None:
                self._device = scalar.device
            elif scalar.device != self._device:
                raise RuntimeError(
                    "DeviceMetricAccumulator received metrics from multiple devices"
                )
            return scalar
        return torch.as_tensor(value, dtype=torch.float32, device=self._device)


__all__ = ["DeviceMetricAccumulator"]
