from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from engine.stage_scheduler import StageScheduler
from utils import LR_SCOPE_DEFAULT, build_optimizer_param_groups


class _ToyStageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aux_prealign = torch.nn.Linear(2, 2, bias=False)
        self.spmf_fusion20 = torch.nn.Linear(2, 2, bias=False)
        self.structure_branch10 = torch.nn.Linear(2, 2, bias=False)
        self.decoder = torch.nn.Linear(2, 1, bias=False)
        self.frozen_backbone = torch.nn.Linear(2, 2, bias=False)
        self.frozen_backbone.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aligned = self.aux_prealign(x)
        fused = self.spmf_fusion20(aligned)
        return self.decoder(fused)


def _make_stages() -> list[dict[str, object]]:
    return [
        {
            "epochs": [1, 2],
            "freeze_modules": ["spmf_fusion20", "structure_branch10"],
            "loss": ["ce"],
        },
        {
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


def _make_stage_lr_trainer(model: torch.nn.Module) -> SimpleNamespace:
    optimizer = torch.optim.SGD(
        build_optimizer_param_groups(
            model,
            weight_decay=0.1,
            base_lr=0.01,
            lr_module_paths=["aux_prealign", "spmf_fusion20", "structure_branch10"],
        ),
        lr=0.01,
        momentum=0.9,
    )
    return SimpleNamespace(
        model=model,
        epoch=1,
        cfg={"class_weights": [1.0, 2.0]},
        device=torch.device("cpu"),
        criterion=None,
        class_weights=None,
        optimizer=optimizer,
        scheduler=None,
    )


def _scope_lrs(trainer: SimpleNamespace) -> dict[str, set[float]]:
    result: dict[str, set[float]] = {}
    for group in trainer.optimizer.param_groups:
        scope = str(group.get("lr_scope", LR_SCOPE_DEFAULT))
        result.setdefault(scope, set()).add(float(group["lr"]))
    return result


def _make_lr_stages() -> list[dict[str, object]]:
    return [
        {
            "epochs": [1, 2],
            "freeze_modules": [],
            "loss": ["ce"],
            "default_lr": 0.01,
            "module_lrs": {
                "spmf_fusion20": 0.001,
                "structure_branch10": 0.001,
            },
        },
        {
            "epochs": [3, 4],
            "freeze_modules": [],
            "loss": ["ce"],
            "default_lr": 0.01,
            "module_lrs": {
                "aux_prealign": 0.001,
            },
        },
    ]


class StageSchedulerTest(unittest.TestCase):
    def test_stage1_freezes_configured_modules(self) -> None:
        model = _ToyStageModel()
        trainer = _make_trainer(model)
        scheduler = StageScheduler(model, _make_stages())

        scheduler.apply(trainer)

        self.assertFalse(any(param.requires_grad for param in model.spmf_fusion20.parameters()))
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

        self.assertTrue(all(param.requires_grad for param in model.spmf_fusion20.parameters()))
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
        self.assertIsNone(model.spmf_fusion20.weight.grad)
        self.assertFalse(model.spmf_fusion20.weight.requires_grad)

    def test_unknown_module_path_raises(self) -> None:
        model = _ToyStageModel()
        trainer = _make_trainer(model)
        stages = [
            {
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
                "epochs": [1, 2],
                "freeze_modules": [],
                "loss": ["ce"],
            },
            {
                "epochs": [2, 3],
                "freeze_modules": [],
                "loss": ["ce"],
            },
        ]
        scheduler = StageScheduler(_ToyStageModel(), stages)

        with self.assertRaisesRegex(ValueError, "Multiple training stages"):
            scheduler.resolve_stage(2)

    def test_stage_lr_overrides_selected_module_scopes(self) -> None:
        model = _ToyStageModel()
        trainer = _make_stage_lr_trainer(model)
        scheduler = StageScheduler(model, _make_lr_stages())

        scheduler.apply(trainer)

        lrs = _scope_lrs(trainer)
        self.assertEqual(lrs[LR_SCOPE_DEFAULT], {0.01})
        self.assertEqual(lrs["aux_prealign"], {0.01})
        self.assertEqual(lrs["spmf_fusion20"], {0.001})
        self.assertEqual(lrs["structure_branch10"], {0.001})

        trainer.epoch = 3
        scheduler.apply(trainer)

        lrs = _scope_lrs(trainer)
        self.assertEqual(lrs[LR_SCOPE_DEFAULT], {0.01})
        self.assertEqual(lrs["aux_prealign"], {0.001})
        self.assertEqual(lrs["spmf_fusion20"], {0.01})
        self.assertEqual(lrs["structure_branch10"], {0.01})

    def test_stage_scheduler_accepts_legacy_spmf_module_paths(self) -> None:
        model = _ToyStageModel()
        trainer = _make_stage_lr_trainer(model)
        scheduler = StageScheduler(
            model,
            [
                {
                    "epochs": [1, 1],
                    "freeze_modules": ["spmf20"],
                    "loss": ["ce"],
                    "module_lrs": {"spmf20": 0.001},
                }
            ],
        )

        scheduler.apply(trainer)

        self.assertFalse(any(param.requires_grad for param in model.spmf_fusion20.parameters()))
        self.assertEqual(_scope_lrs(trainer)["spmf_fusion20"], {0.001})

    def test_stage_lr_uses_multistep_scheduler_scale(self) -> None:
        model = _ToyStageModel()
        optimizer = torch.optim.SGD(
            build_optimizer_param_groups(
                model,
                weight_decay=0.1,
                base_lr=0.01,
                lr_module_paths=["aux_prealign", "spmf_fusion20", "structure_branch10"],
            ),
            lr=0.01,
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[1],
            gamma=0.1,
        )
        optimizer.step()
        lr_scheduler.step()
        trainer = SimpleNamespace(
            model=model,
            epoch=1,
            cfg={"class_weights": [1.0, 2.0]},
            device=torch.device("cpu"),
            criterion=None,
            class_weights=None,
            optimizer=optimizer,
            scheduler=lr_scheduler,
        )
        scheduler = StageScheduler(model, _make_lr_stages())

        scheduler.apply(trainer)

        lrs = _scope_lrs(trainer)
        self.assertEqual(lrs[LR_SCOPE_DEFAULT], {0.001})
        self.assertEqual(lrs["spmf_fusion20"], {0.0001})

    def test_stage_lr_rejects_invalid_values_and_modules(self) -> None:
        model = _ToyStageModel()

        with self.assertRaisesRegex(ValueError, "must be positive"):
            StageScheduler(
                model,
                [
                    {
                        "epochs": [1, 1],
                        "freeze_modules": [],
                        "loss": ["ce"],
                        "default_lr": 0.0,
                    }
                ],
            )
        with self.assertRaisesRegex(TypeError, "module_lrs keys must be non-empty strings"):
            StageScheduler(
                model,
                [
                    {
                        "epochs": [1, 1],
                        "freeze_modules": [],
                        "loss": ["ce"],
                        "module_lrs": {"": 0.001},
                    }
                ],
            )
        with self.assertRaisesRegex(AttributeError, "missing"):
            StageScheduler(
                model,
                [
                    {
                        "epochs": [1, 1],
                        "freeze_modules": [],
                        "loss": ["ce"],
                        "module_lrs": {"missing": 0.001},
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
