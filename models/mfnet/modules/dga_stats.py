from __future__ import annotations

from typing import Any

import torch


def record_dga_intermediate_stats(
    module: Any,
    *,
    alpha_gate_name: str,
    alpha_gate: torch.Tensor,
    beta_gate_name: str,
    beta_gate: torch.Tensor,
    alpha_injection: torch.Tensor,
    beta_injection: torch.Tensor,
    alpha_main: torch.Tensor,
    beta_main: torch.Tensor,
) -> None:
    stats = getattr(module, "intermediate_stats", None)
    if stats is None:
        return

    prefix = str(getattr(module, "intermediate_stats_prefix", "dga")).strip("/")
    stats.record_mean_std(f"{prefix}/{alpha_gate_name}", alpha_gate)
    stats.record_mean_std(f"{prefix}/{beta_gate_name}", beta_gate)
    stats.record_norm_ratio(f"{prefix}/alpha_injection_ratio", alpha_injection, alpha_main)
    stats.record_norm_ratio(f"{prefix}/beta_injection_ratio", beta_injection, beta_main)
    stats.record_norm(f"{prefix}/alpha_injection_norm", alpha_injection)
    stats.record_norm(f"{prefix}/beta_injection_norm", beta_injection)
    stats.record_norm(f"{prefix}/alpha_main_norm", alpha_main)
    stats.record_norm(f"{prefix}/beta_main_norm", beta_main)


__all__ = ["record_dga_intermediate_stats"]
