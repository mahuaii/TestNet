from __future__ import annotations

from typing import Any

import torch

from .dga10 import DGABlock10, _validate_pair as _validate_pair_dga10
from .dga20 import DGABlock20, _validate_pair as _validate_pair_dga20


def _norm(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(value.detach())


def _norm_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(denominator.dtype).eps
    return _norm(numerator) / _norm(denominator).clamp_min(eps)


def _record_contribution_stats(
    module: Any,
    *,
    alpha_injection: torch.Tensor,
    beta_injection: torch.Tensor,
    alpha_main: torch.Tensor,
    beta_main: torch.Tensor,
) -> None:
    with torch.no_grad():
        module.last_dga_stats = {
            "alpha_injection_ratio": _norm_ratio(alpha_injection, alpha_main),
            "beta_injection_ratio": _norm_ratio(beta_injection, beta_main),
            "alpha_injection_norm": _norm(alpha_injection),
            "beta_injection_norm": _norm(beta_injection),
            "alpha_main_norm": _norm(alpha_main),
            "beta_main_norm": _norm(beta_main),
        }


class DGABlock10ContributionStats(DGABlock10):
    def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_pair_dga10(rgb, aux, self.channels)
        rgb_norm = self.norm_rgb(rgb)
        aux_norm = self.norm_aux(aux)
        difference = rgb_norm - aux_norm

        aux_to_rgb_message = self.message_aux_to_rgb(torch.cat([aux_norm, difference], dim=1))
        rgb_to_aux_message = self.message_rgb_to_aux(torch.cat([rgb_norm, -difference], dim=1))
        rgb_gate = self.gate_rgb(torch.cat([rgb_norm, difference], dim=1))
        aux_gate = self.gate_aux(torch.cat([aux_norm, -difference], dim=1))

        rgb_injection = self.alpha * rgb_gate * aux_to_rgb_message
        aux_injection = self.beta * aux_gate * rgb_to_aux_message
        _record_contribution_stats(
            self,
            alpha_injection=rgb_injection,
            beta_injection=aux_injection,
            alpha_main=rgb,
            beta_main=aux,
        )
        return rgb + rgb_injection, aux + aux_injection


class DGABlock20ContributionStats(DGABlock20):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_pair_dga20(x, y, self.channels)
        x_norm = self.norm_x(x)
        y_norm = self.norm_y(y)
        difference = x_norm - y_norm

        y_to_x_message = self.message_y_to_x(torch.cat([y_norm, difference], dim=1))
        x_to_y_message = self.message_x_to_y(torch.cat([x_norm, -difference], dim=1))
        x_gate = self.gate_x(torch.cat([x_norm, difference], dim=1))
        y_gate = self.gate_y(torch.cat([y_norm, -difference], dim=1))

        x_injection = self.alpha * x_gate * y_to_x_message
        y_injection = self.beta * y_gate * x_to_y_message
        _record_contribution_stats(
            self,
            alpha_injection=x_injection,
            beta_injection=y_injection,
            alpha_main=x,
            beta_main=y,
        )
        return x + x_injection, y + y_injection


__all__ = [
    "DGABlock10ContributionStats",
    "DGABlock20ContributionStats",
]
