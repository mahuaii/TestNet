from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from engine.stage_scheduler import StageScheduler


class _ToyStageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aux_prealign = torch.nn.Linear(2, 2, bias=False)
        self.spmf20 = torch.nn.Linear(2, 2, bias=False)
        self.structure_branch10 = torch.nn.Linear(2, 2, bias=False)
        self.decoder = torch.nn.Linear(2, 1, bias=False)
        self.frozen_backbone = torch.nn.Linear(2, 2, bias=False)
        self.frozen_backbone.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aligned = self.aux_prealign(x)
        fused = self.spmf20(aligned)
        return self.decoder(fused)


def _make_stages() -> list[dict[str, object]]:
    return [
        {
            "name": "prealign",
            "epochs": [1, 2],
            "freeze_modules": ["spmf20", "structure_branch10"],
            "loss": ["ce"],
        },
        {
            "name": "joint",
            "epochs": [3, 4],
            "freeze_modules": [],
            "loss": ["ce", "lovasz"],
            "loss_weights": {"lovasz": 0.3},
        },
    ]


def _make_trainer(model: torch.nn.Module) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        epoch=1,
        cfg={"class_weights": [1.0, 2.0]},
        device=torch.device("cpu"),
        criterion=None,
        class_weights=None,
    )


class StageSchedulerTest(unittest.TestCase):
    def test_stage1_freezes_configured_modules(self) -> None:
        model = _ToyStageModel()
        trainer = _make_trainer(model)
        scheduler = StageScheduler(model, _make_stages())

        scheduler.apply(trainer)

        self.assertFalse(any(param.requires_grad for param in model.spmf20.parameters()))
        self.assertFalse(any(param.requires_grad for param in model.structure_branch10.parameters()))
        self.assertTrue(all(param.requires_grad for param in model.aux_prealign.parameters()))
        self.assertTrue(all(param.requires_grad for param in model.decoder.parameters()))
        self.assertFalse(any(param.requires_grad for param in model.frozen_backbone.parameters()))
        self.assertEqual([loss.name for loss in trainer.criterion.losses], ["ce"])

    def test_later_stage_restores_baseline_trainability_and_switches_loss(self) -> None:
        model = _ToyStageModel()
        trainer = _make_trainer(model)
        scheduler = StageScheduler(model, _make_stages())
        trainer.epoch = 1
        scheduler.apply(trainer)

        trainer.epoch = 3
        scheduler.apply(trainer)

        self.assertTrue(all(param.requires_grad for param in model.spmf20.parameters()))
        self.assertTrue(all(param.requires_grad for param in model.structure_branch10.parameters()))
        self.assertFalse(any(param.requires_grad for param in model.frozen_backbone.parameters()))
        self.assertEqual([loss.name for loss in trainer.criterion.losses], ["ce", "lovasz"])
        self.assertEqual(float(trainer.criterion.losses[1].weight), 0.3)

    def test_stage_freeze_keeps_gradient_path_to_earlier_modules(self) -> None:
        model = _ToyStageModel()
        trainer = _make_trainer(model)
        scheduler = StageScheduler(model, _make_stages())

        scheduler.apply(trainer)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()

        self.assertIsNotNone(model.aux_prealign.weight.grad)
        self.assertIsNone(model.spmf20.weight.grad)
        self.assertFalse(model.spmf20.weight.requires_grad)

    def test_unknown_module_path_raises(self) -> None:
        model = _ToyStageModel()
        trainer = _make_trainer(model)
        stages = [
            {
                "name": "bad",
                "epochs": [1, 1],
                "freeze_modules": ["missing"],
                "loss": ["ce"],
            }
        ]
        scheduler = StageScheduler(model, stages)

        with self.assertRaisesRegex(AttributeError, "missing"):
            scheduler.apply(trainer)

    def test_epoch_without_stage_raises(self) -> None:
        scheduler = StageScheduler(_ToyStageModel(), _make_stages())

        with self.assertRaisesRegex(ValueError, "No training stage"):
            scheduler.resolve_stage(5)

    def test_overlapping_stage_raises(self) -> None:
        stages = [
            {
                "name": "first",
                "epochs": [1, 2],
                "freeze_modules": [],
                "loss": ["ce"],
            },
            {
                "name": "second",
                "epochs": [2, 3],
                "freeze_modules": [],
                "loss": ["ce"],
            },
        ]
        scheduler = StageScheduler(_ToyStageModel(), stages)

        with self.assertRaisesRegex(ValueError, "Multiple training stages"):
            scheduler.resolve_stage(2)

    def test_stage_without_name_raises(self) -> None:
        stages = [
            {
                "epochs": [1, 1],
                "freeze_modules": [],
                "loss": ["ce"],
            }
        ]

        with self.assertRaisesRegex(KeyError, "name"):
            StageScheduler(_ToyStageModel(), stages)


if __name__ == "__main__":
    unittest.main()
