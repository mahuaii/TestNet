from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch


class IntermediateStatsRecorder:
    """
    Collect scalar statistics from arbitrary model internals during a forward pass.

    The recorder stores detached CPU floats only. It never keeps feature tensors, so
    inserting calls in model code does not retain autograd graphs or large activations.
    """

    def __init__(self, *, enabled: bool = True, prefix: str = "") -> None:
        self.enabled = bool(enabled)
        self.prefix = prefix.strip("/")
        self._stats: dict[str, float] = {}

    def clear(self) -> None:
        self._stats.clear()

    def snapshot(self, *, reset: bool = False) -> dict[str, float]:
        stats = dict(self._stats)
        if reset:
            self.clear()
        return stats

    def record_scalar(self, name: str, value: Any) -> None:
        if not self.enabled:
            return
        self._stats[self._key(name)] = self._to_float(value)

    def record_mean_std(self, name: str, value: torch.Tensor) -> None:
        if not self.enabled:
            return
        tensor = self._numeric_tensor(value)
        with torch.no_grad():
            self.record_scalar(f"{name}_mean", tensor.mean())
            self.record_scalar(f"{name}_std", tensor.std(unbiased=False))

    def record_norm(self, name: str, value: torch.Tensor) -> None:
        if not self.enabled:
            return
        tensor = self._numeric_tensor(value)
        with torch.no_grad():
            self.record_scalar(name, torch.linalg.vector_norm(tensor))

    def record_norm_ratio(
        self,
        name: str,
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        *,
        eps: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        numerator_tensor = self._numeric_tensor(numerator)
        denominator_tensor = self._numeric_tensor(denominator)
        with torch.no_grad():
            denominator_norm = torch.linalg.vector_norm(denominator_tensor)
            clamp_eps = self._default_eps(denominator_tensor) if eps is None else float(eps)
            ratio = torch.linalg.vector_norm(numerator_tensor) / denominator_norm.clamp_min(clamp_eps)
            self.record_scalar(name, ratio)

    @contextmanager
    def disabled(self) -> Iterator[None]:
        was_enabled = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = was_enabled

    def _key(self, name: str) -> str:
        normalized_name = name.strip("/")
        if not normalized_name:
            raise ValueError("stat name must not be empty")
        if not self.prefix:
            return normalized_name
        return f"{self.prefix}/{normalized_name}"

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"expected a scalar tensor, got shape {tuple(value.shape)}")
            return float(value.detach().cpu())
        return float(value)

    @staticmethod
    def _numeric_tensor(value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"expected torch.Tensor, got {type(value).__name__}")
        tensor = value.detach()
        if not (tensor.is_floating_point() or torch.is_complex(tensor)):
            tensor = tensor.float()
        return tensor

    @staticmethod
    def _default_eps(value: torch.Tensor) -> float:
        if value.is_floating_point():
            return float(torch.finfo(value.dtype).eps)
        if torch.is_complex(value):
            return float(torch.finfo(value.real.dtype).eps)
        return float(torch.finfo(torch.float32).eps)


__all__ = ["IntermediateStatsRecorder"]
