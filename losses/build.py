from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

from .combined_loss import CombinedLoss
from .loss_modules import (
    BoundaryLossModule,
    CrossEntropyLossModule,
    LossModule,
    LovaszLossModule,
)


def build_loss(
    cfg: list[str] | None,
    *,
    weights: Mapping[str, float] | None = None,
    class_weights: list[float] | None = None,
) -> CombinedLoss:
    loss_names = ["ce"] if cfg is None else cfg
    if not isinstance(loss_names, list) or not all(isinstance(name, str) for name in loss_names):
        raise TypeError("Loss configuration must be a list of loss names.")
    if not loss_names:
        raise ValueError("At least one loss must be configured.")
    if len(loss_names) != len(set(name.strip().lower() for name in loss_names)):
        raise ValueError("Loss configuration contains duplicate loss names.")

    default_weights = {
        "ce": 1.0,
        "lovasz": 0.2,
        "boundary": 0.05,
    }
    configured_weights = dict(weights or {})
    unknown_weight_names = configured_weights.keys() - default_weights.keys()
    if unknown_weight_names:
        unknown = ", ".join(sorted(unknown_weight_names))
        raise ValueError(f"Unknown loss weight name(s): {unknown}.")
    for name, weight in configured_weights.items():
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise TypeError(f"Loss weight {name!r} must be a real number.")
        if float(weight) < 0.0:
            raise ValueError(f"Loss weight {name!r} must be non-negative.")

    resolved_weights = {
        name: float(configured_weights.get(name, default_weight))
        for name, default_weight in default_weights.items()
    }
    builders = {
        "ce": lambda: CrossEntropyLossModule(
            class_weights=class_weights,
            weight=resolved_weights["ce"],
        ),
        "lovasz": lambda: LovaszLossModule(
            weight=resolved_weights["lovasz"],
        ),
        "boundary": lambda: BoundaryLossModule(
            weight=resolved_weights["boundary"],
        ),
    }

    losses: list[LossModule] = []
    for configured_name in loss_names:
        loss_name = configured_name.strip().lower()
        try:
            builder = builders[loss_name]
        except KeyError as exc:
            available = ", ".join(sorted(builders))
            raise ValueError(
                f"Unsupported loss type: {loss_name!r}. Available: {available}."
            ) from exc
        losses.append(builder())
    return CombinedLoss(losses)


__all__ = ["build_loss"]
