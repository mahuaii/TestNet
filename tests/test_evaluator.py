from __future__ import annotations

import unittest

import torch

from engine import Evaluator


class EvaluatorTest(unittest.TestCase):
    def test_evaluate_matches_original_mfnet_metrics(self) -> None:
        outputs = [
            {
                "pred": torch.tensor([0, 1, 1, 3, 5, 5, 0, 0], dtype=torch.long),
                "target": torch.tensor([0, 1, 2, 3, 4, 5, 6, 255], dtype=torch.long),
            }
        ]

        metrics = Evaluator().evaluate(
            outputs,
            num_classes=6,
            metric_classes=5,
        )

        for key in ["MIoU", "accuracy", "F1Score", "kappa"]:
            self.assertIn(key, metrics)
        for key in [
            "confusion_matrix",
            "pixels_processed",
            "class_names",
            "per_class_accuracy",
            "per_class_f1",
            "per_class_iou",
        ]:
            self.assertIn(key, metrics)
        self.assertAlmostEqual(metrics["MIoU"], 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 100.0 * 4.0 / 6.0)
        self.assertAlmostEqual(metrics["F1Score"], (1.0 + (2.0 / 3.0) + 0.0 + 1.0 + 0.0) / 5.0)
        self.assertAlmostEqual(metrics["kappa"], 0.6)
        self.assertEqual(metrics["pixels_processed"], 6)
        self.assertEqual(metrics["class_names"], ["roads", "buildings", "low veg.", "trees", "cars", "clutter"])
        self.assertEqual(metrics["confusion_matrix"].shape, (6, 6))
        self.assertEqual(metrics["per_class_accuracy"].shape, (6,))
        self.assertEqual(metrics["per_class_f1"].shape, (6,))
        self.assertEqual(metrics["per_class_iou"].shape, (6,))

    def test_evaluate_uses_configured_metric_class_count(self) -> None:
        outputs = [
            {
                "pred": torch.tensor([0, 1, 1, 3, 5, 5], dtype=torch.long),
                "target": torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
            }
        ]

        metrics = Evaluator().evaluate(
            outputs,
            num_classes=6,
            metric_classes=5,
        )

        self.assertAlmostEqual(metrics["MIoU"], 0.5)
        self.assertAlmostEqual(metrics["F1Score"], (1.0 + (2.0 / 3.0) + 0.0 + 1.0 + 0.0) / 5.0)

    def test_evaluate_matches_original_nanmean_for_missing_f1_classes(self) -> None:
        outputs = [
            {
                "pred": torch.tensor([0, 1, 1], dtype=torch.long),
                "target": torch.tensor([0, 1, 2], dtype=torch.long),
            }
        ]

        metrics = Evaluator().evaluate(
            outputs,
            num_classes=6,
            metric_classes=5,
        )

        self.assertAlmostEqual(metrics["F1Score"], (1.0 + (2.0 / 3.0) + 0.0) / 3.0)
        self.assertTrue(torch.isnan(torch.tensor(metrics["per_class_f1"][3])))
        self.assertTrue(torch.isnan(torch.tensor(metrics["per_class_f1"][4])))

    def test_evaluate_accepts_trainer_context_from_base_validate(self) -> None:
        class TrainerStub:
            cfg = {"num_classes": 6}

        outputs = [
            {
                "pred": torch.tensor([0, 1, 1, 3, 5, 5], dtype=torch.long),
                "target": torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
            }
        ]

        metrics = Evaluator().evaluate(outputs, trainer=TrainerStub())

        self.assertAlmostEqual(metrics["MIoU"], 0.5)


if __name__ == "__main__":
    unittest.main()
