from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from utils import safe_path_component
from utils.logger import Logger


_SAMPLE_INTERVAL = 100
_RATIO_EPS = 1e-12


@dataclass(frozen=True)
class _SampledParam:
    param: torch.nn.Parameter
    before: torch.Tensor
    param_group: str
    module_group: str


@dataclass
class _NormAccumulator:
    weight_sq: torch.Tensor | None = None
    grad_sq: torch.Tensor | None = None
    update_sq: torch.Tensor | None = None

    def add(
        self,
        *,
        before: torch.Tensor,
        grad: torch.Tensor,
        update: torch.Tensor,
    ) -> None:
        weight_sq = before.detach().float().pow(2).sum()
        grad_sq = grad.detach().float().pow(2).sum()
        update_sq = update.detach().float().pow(2).sum()
        if self.weight_sq is None:
            self.weight_sq = weight_sq
            self.grad_sq = grad_sq
            self.update_sq = update_sq
            return
        self.weight_sq = self.weight_sq + weight_sq
        self.grad_sq = self.grad_sq + grad_sq
        self.update_sq = self.update_sq + update_sq

    def snapshot(self) -> dict[str, float]:
        if self.weight_sq is None or self.grad_sq is None or self.update_sq is None:
            raise RuntimeError("Cannot snapshot empty norm accumulator.")
        weight_norm = float(torch.sqrt(self.weight_sq).item())
        grad_norm = float(torch.sqrt(self.grad_sq).item())
        update_norm = float(torch.sqrt(self.update_sq).item())
        return {
            "weight_norm": weight_norm,
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "update_ratio": update_norm / (weight_norm + _RATIO_EPS),
        }


class ModuleNormMonitor:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        logger: Logger,
    ) -> None:
        self.optimizer = optimizer
        self.logger = logger
        self._param_names = {
            id(param): name
            for name, param in model.named_parameters()
        }
        self._samples: list[_SampledParam] = []

    def before_step(self, *, next_global_step: int) -> bool:
        self._samples = []
        if next_global_step % _SAMPLE_INTERVAL != 0:
            return False

        with torch.no_grad():
            for group_index, group in enumerate(self.optimizer.param_groups):
                param_group = self._param_group_tag(index=group_index, group=group)
                for param in group["params"]:
                    if not param.requires_grad or param.grad is None:
                        continue
                    name = self._param_names[id(param)]
                    self._samples.append(
                        _SampledParam(
                            param=param,
                            before=param.detach().clone(),
                            param_group=param_group,
                            module_group=self._module_group(name),
                        )
                    )
        return bool(self._samples)

    def after_step(self, *, global_step: int) -> None:
        if not self._samples:
            return

        param_group_accumulators: dict[str, _NormAccumulator] = {}
        module_group_accumulators: dict[str, _NormAccumulator] = {}
        with torch.no_grad():
            for sample in self._samples:
                grad = sample.param.grad
                if grad is None:
                    continue
                update = sample.param.detach() - sample.before
                self._accumulator(
                    param_group_accumulators,
                    sample.param_group,
                ).add(
                    before=sample.before,
                    grad=grad,
                    update=update,
                )
                self._accumulator(
                    module_group_accumulators,
                    sample.module_group,
                ).add(
                    before=sample.before,
                    grad=grad,
                    update=update,
                )

        self.logger.log_module_norm_scalars(
            global_step=global_step,
            param_group_stats=self._snapshot(param_group_accumulators),
            module_group_stats=self._snapshot(module_group_accumulators),
        )
        self._samples = []

    @staticmethod
    def _accumulator(
        accumulators: dict[str, _NormAccumulator],
        key: str,
    ) -> _NormAccumulator:
        accumulator = accumulators.get(key)
        if accumulator is None:
            accumulator = _NormAccumulator()
            accumulators[key] = accumulator
        return accumulator

    @staticmethod
    def _snapshot(
        accumulators: Mapping[str, _NormAccumulator],
    ) -> dict[str, dict[str, float]]:
        return {
            key: accumulator.snapshot()
            for key, accumulator in sorted(accumulators.items())
        }

    @staticmethod
    def _param_group_tag(*, index: int, group: Mapping[str, object]) -> str:
        group_name = str(group.get("group_name", f"group_{index}"))
        safe_group_name = safe_path_component(group_name, f"group_{index}")
        return f"{index}_{safe_group_name}"

    @staticmethod
    def _module_group(name: str) -> str:
        if name.startswith("image_encoder."):
            if "Adapter" in name:
                return "image_encoder_adapter"
            if "lora_" in name:
                return "image_encoder_lora"
            return "image_encoder_other"
        return name.split(".", maxsplit=1)[0]


__all__ = ["ModuleNormMonitor"]
