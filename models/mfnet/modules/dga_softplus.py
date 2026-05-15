from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .dga10 import DGABlock10
from .dga20 import DGABlock20


def _inv_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError(f"Expected softplus scale initial value to be positive, got {value}.")
    return math.log(math.expm1(float(value)))


def _reset_scale_parameters_to_softplus_raw(block: DGABlock10 | DGABlock20) -> None:
    raw_init = _inv_softplus(float(block.init_scale))
    block.alpha = torch.nn.Parameter(torch.full_like(block.alpha.detach(), raw_init))
    block.beta = torch.nn.Parameter(torch.full_like(block.beta.detach(), raw_init))


class DGABlock10Softplus(DGABlock10):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        _reset_scale_parameters_to_softplus_raw(self)

    def effective_alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha)

    def effective_beta(self) -> torch.Tensor:
        return F.softplus(self.beta)


class DGABlock20Softplus(DGABlock20):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        _reset_scale_parameters_to_softplus_raw(self)

    def effective_alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha)

    def effective_beta(self) -> torch.Tensor:
        return F.softplus(self.beta)


__all__ = ["DGABlock10Softplus", "DGABlock20Softplus"]
