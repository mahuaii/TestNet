from __future__ import annotations

import unittest

import torch

from losses import CombinedLoss, build_loss, make_boundary_target
from utils import DataUtils


class LossTest(unittest.TestCase):
    def test_build_ce_loss_matches_existing_cross_entropy(self) -> None:
        logits = torch.tensor(
            [
                [
                    [[2.0, 0.0], [0.5, 1.0]],
                    [[0.0, 2.0], [1.5, 0.0]],
                ]
            ],
            requires_grad=True,
        )
        target = torch.tensor([[[0, 1], [1, 255]]], dtype=torch.long)
        class_weights = [1.0, 2.0]

        criterion = build_loss(
            ["ce"],
            class_weights=class_weights,
        )
        total, items = criterion(logits, target)
        expected = DataUtils.cross_entropy_filtered(
            logits=logits,
            target=target,
            weight=torch.tensor(class_weights),
            ignore_label=255,
        )

        self.assertIsInstance(criterion, CombinedLoss)
        torch.testing.assert_close(total, expected)
        torch.testing.assert_close(items["ce"], expected)

    def test_ce_lovasz_adds_weighted_lovasz_loss(self) -> None:
        logits = torch.randn(2, 3, 4, 4, requires_grad=True)
        target = torch.randint(0, 3, (2, 4, 4))
        criterion = build_loss(
            ["ce", "lovasz"],
            weights={"ce": 0.5, "lovasz": 0.3, "boundary": 0.05},
        )

        total, items = criterion(logits, target)

        self.assertEqual(set(items), {"ce", "lovasz"})
        torch.testing.assert_close(total, 0.5 * items["ce"] + 0.3 * items["lovasz"])
        total.backward()
        self.assertIsNotNone(logits.grad)

    def test_boundary_loss_uses_boundary_logits_and_ignores_invalid_pixels(self) -> None:
        segmentation_logits = torch.randn(1, 2, 3, 4, requires_grad=True)
        boundary_logits = torch.zeros(1, 1, 3, 4, requires_grad=True)
        target = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [255, 255, 1, 1]]],
            dtype=torch.long,
        )
        criterion = build_loss(["ce", "boundary"])

        total, items = criterion(
            {
                "logits": segmentation_logits,
                "boundary_logits": boundary_logits,
            },
            target,
        )

        self.assertEqual(
            set(items),
            {"ce", "boundary", "boundary_bce", "boundary_dice"},
        )
        torch.testing.assert_close(total, items["ce"] + 0.05 * items["boundary"])
        total.backward()
        self.assertIsNotNone(segmentation_logits.grad)
        self.assertIsNotNone(boundary_logits.grad)

        boundary_target, valid_mask = make_boundary_target(target)
        self.assertEqual(boundary_target.shape, (1, 1, 3, 4))
        self.assertEqual(valid_mask.shape, (1, 1, 3, 4))
        self.assertEqual(float(valid_mask[0, 0, 2, 0]), 0.0)

    def test_boundary_loss_requires_boundary_logits(self) -> None:
        criterion = build_loss(["ce", "boundary"])
        logits = torch.randn(1, 2, 2, 2)
        target = torch.zeros(1, 2, 2, dtype=torch.long)

        with self.assertRaisesRegex(KeyError, "boundary_logits"):
            criterion(logits, target)

    def test_build_loss_rejects_unknown_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported loss type"):
            build_loss(["unknown"])

    def test_build_loss_rejects_duplicate_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate loss names"):
            build_loss(["ce", "CE"])

    def test_build_loss_rejects_empty_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one loss"):
            build_loss([])

    def test_build_loss_rejects_unknown_weight_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown loss weight"):
            build_loss(["ce"], weights={"dice": 1.0})

    def test_build_loss_rejects_negative_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_loss(["ce"], weights={"ce": -1.0})


if __name__ == "__main__":
    unittest.main()
