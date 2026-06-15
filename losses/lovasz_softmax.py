from __future__ import annotations

import torch


def lovasz_softmax_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    *,
    classes: str = "present",
    ignore_index: int = 255,
) -> torch.Tensor:
    if probabilities.ndim != 4:
        raise ValueError(
            "Expected probabilities with shape [B, C, H, W], "
            f"got {tuple(probabilities.shape)}."
        )
    if target.shape != (probabilities.shape[0], *probabilities.shape[-2:]):
        raise ValueError(
            "Expected target shape to match probability spatial dimensions, "
            f"got probabilities {tuple(probabilities.shape)} and target {tuple(target.shape)}."
        )
    if classes not in {"all", "present"}:
        raise ValueError(f"Expected classes to be 'all' or 'present', got {classes!r}.")

    probabilities_flat, target_flat = _flatten_probabilities(
        probabilities,
        target,
        ignore_index=ignore_index,
    )
    if target_flat.numel() == 0:
        return probabilities.sum() * 0.0

    losses: list[torch.Tensor] = []
    for class_index in range(probabilities.shape[1]):
        foreground = (target_flat == class_index).to(probabilities_flat.dtype)
        if classes == "present" and not torch.any(foreground):
            continue

        class_probabilities = probabilities_flat[:, class_index]
        errors = (foreground - class_probabilities).abs()
        errors_sorted, permutation = torch.sort(errors, descending=True)
        foreground_sorted = foreground[permutation]
        losses.append(torch.dot(errors_sorted, _lovasz_gradient(foreground_sorted)))

    if not losses:
        return probabilities.sum() * 0.0
    return torch.stack(losses).mean()


def _flatten_probabilities(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities_flat = probabilities.permute(0, 2, 3, 1).contiguous().view(-1, probabilities.shape[1])
    target_flat = target.contiguous().view(-1)
    valid_mask = (
        (target_flat >= 0)
        & (target_flat < probabilities.shape[1])
        & (target_flat != ignore_index)
    )
    return probabilities_flat[valid_mask], target_flat[valid_mask]


def _lovasz_gradient(foreground_sorted: torch.Tensor) -> torch.Tensor:
    num_pixels = foreground_sorted.numel()
    foreground_total = foreground_sorted.sum()
    intersection = foreground_total - foreground_sorted.cumsum(dim=0)
    union = foreground_total + (1.0 - foreground_sorted).cumsum(dim=0)
    gradient = 1.0 - intersection / union.clamp_min(torch.finfo(foreground_sorted.dtype).eps)
    if num_pixels > 1:
        gradient = torch.cat((gradient[:1], gradient[1:] - gradient[:-1]))
    return gradient


__all__ = ["lovasz_softmax_loss"]
