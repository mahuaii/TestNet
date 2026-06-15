from __future__ import annotations

import torch
import torch.nn.functional as F


def boundary_loss(
    boundary_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int = 255,
    pos_weight: torch.Tensor | None = None,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    boundary_width: int = 1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if boundary_logits.ndim != 4 or boundary_logits.shape[1] != 1:
        raise ValueError(
            "Expected boundary_logits with shape [B, 1, H, W], "
            f"got {tuple(boundary_logits.shape)}."
        )
    if target.ndim != 3 or target.shape[0] != boundary_logits.shape[0]:
        raise ValueError(
            "Expected target with shape [B, H, W] and matching batch size, "
            f"got {tuple(target.shape)}."
        )
    if boundary_width < 1:
        raise ValueError(f"Expected boundary_width to be at least 1, got {boundary_width}.")

    if tuple(boundary_logits.shape[-2:]) != tuple(target.shape[-2:]):
        boundary_logits = F.interpolate(
            boundary_logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    boundary_target, valid_mask = make_boundary_target(
        target,
        ignore_index=ignore_index,
        boundary_width=boundary_width,
    )
    valid_count = valid_mask.sum()
    if valid_count == 0:
        zero = boundary_logits.sum() * 0.0
        return zero, {"boundary_bce": zero, "boundary_dice": zero}

    bce_map = F.binary_cross_entropy_with_logits(
        boundary_logits,
        boundary_target,
        pos_weight=pos_weight,
        reduction="none",
    )
    loss_bce = (bce_map * valid_mask).sum() / valid_count

    boundary_probabilities = torch.sigmoid(boundary_logits) * valid_mask
    masked_target = boundary_target * valid_mask
    intersection = (boundary_probabilities * masked_target).sum()
    denominator = boundary_probabilities.sum() + masked_target.sum()
    loss_dice = 1.0 - (2.0 * intersection + eps) / (denominator + eps)

    total = float(bce_weight) * loss_bce + float(dice_weight) * loss_dice
    return total, {
        "boundary_bce": loss_bce,
        "boundary_dice": loss_dice,
    }


def make_boundary_target(
    target: torch.Tensor,
    *,
    ignore_index: int = 255,
    boundary_width: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.ndim != 3:
        raise ValueError(f"Expected target with shape [B, H, W], got {tuple(target.shape)}.")
    if boundary_width < 1:
        raise ValueError(f"Expected boundary_width to be at least 1, got {boundary_width}.")

    valid = target != ignore_index
    boundary = torch.zeros_like(valid)

    vertical_difference = valid[:, 1:, :] & valid[:, :-1, :] & (target[:, 1:, :] != target[:, :-1, :])
    boundary[:, 1:, :] |= vertical_difference
    boundary[:, :-1, :] |= vertical_difference

    horizontal_difference = valid[:, :, 1:] & valid[:, :, :-1] & (target[:, :, 1:] != target[:, :, :-1])
    boundary[:, :, 1:] |= horizontal_difference
    boundary[:, :, :-1] |= horizontal_difference

    boundary = boundary.unsqueeze(1).to(torch.float32)
    kernel_size = 2 * boundary_width - 1
    if kernel_size > 1:
        boundary = F.max_pool2d(
            boundary,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    invalid = (~valid).unsqueeze(1).to(torch.float32)
    invalid_neighborhood = F.max_pool2d(
        invalid,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    valid_mask = (invalid_neighborhood == 0).to(boundary.dtype)
    return boundary, valid_mask


__all__ = ["boundary_loss", "make_boundary_target"]
