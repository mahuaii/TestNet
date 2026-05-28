from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from engine import MFNetTrainer
from utils import CheckpointManager, TestNetLogger


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


class _Dataset:
    ids = ["1"]
    patch_size = (16, 16)


class _ValLoader:
    dataset = _Dataset()


class _Inferencer:
    def __init__(self) -> None:
        self.call: dict[str, Any] | None = None
        self.outputs = [
            {
                "pred": torch.zeros(2, 2, dtype=torch.long),
                "target": torch.zeros(2, 2, dtype=torch.long),
                "meta": {"tile_id": "1"},
            }
        ]

    def run(
        self,
        model: torch.nn.Module,
        dataset: object,
        device: torch.device,
        stride: int,
        batch_size: int,
        window_size: tuple[int, int],
        num_classes: int,
        input_modals: tuple[str, ...],
        model_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.call = {
            "model": model,
            "dataset": dataset,
            "device": device,
            "stride": stride,
            "batch_size": batch_size,
            "window_size": window_size,
            "num_classes": num_classes,
            "input_modals": input_modals,
            "model_kwargs": model_kwargs,
        }
        return self.outputs


class _Evaluator:
    def __init__(self) -> None:
        self.outputs: list[dict[str, Any]] | None = None
        self.num_classes: int | None = None

    def evaluate(self, outputs: list[dict[str, Any]], num_classes: int) -> dict[str, float]:
        self.outputs = outputs
        self.num_classes = num_classes
        return {"MIoU": 0.7, "accuracy": 88.0, "F1Score": 0.8, "kappa": 0.6}


class MFNetValidateTest(unittest.TestCase):
    def test_validate_uses_whole_tile_inferencer_and_updates_best_miou(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _Model()
            inferencer = _Inferencer()
            evaluator = _Evaluator()
            trainer = MFNetTrainer(
                model=model,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                scheduler=None,
                train_loader=[{"unused": torch.tensor(1)}],
                val_loader=_ValLoader(),
                logger=TestNetLogger(tmpdir),
                evaluator=evaluator,
                inferencer=inferencer,
                device=torch.device("cpu"),
                cfg={
                    "work_dir": tmpdir,
                    "max_epochs": 1,
                    "batch_size": 2,
                    "log_step_interval": 1,
                    "val_epoch_interval": 1,
                    "save_epoch_interval": 0,
                    "save_step_interval": 0,
                    "validation": {"stride": 32},
                    "num_classes": 6,
                },
            )

            trainer.validate()

            assert inferencer.call is not None
            self.assertIs(inferencer.call["model"], trainer.model)
            self.assertIs(inferencer.call["dataset"], trainer.val_loader.dataset)
            self.assertEqual(inferencer.call["device"], torch.device("cpu"))
            self.assertEqual(inferencer.call["stride"], 32)
            self.assertEqual(inferencer.call["batch_size"], 2)
            self.assertEqual(inferencer.call["window_size"], (16, 16))
            self.assertEqual(inferencer.call["num_classes"], 6)
            self.assertEqual(inferencer.call["input_modals"], ("rgb", "dsm"))
            self.assertEqual(inferencer.call["model_kwargs"], {"mode": "Test"})
            self.assertIs(evaluator.outputs, inferencer.outputs)
            self.assertEqual(evaluator.num_classes, 6)
            self.assertTrue(trainer.model.training)
            self.assertEqual(trainer.best_miou, 0.7)
            best_state_path = Path(tmpdir, "best_miou.pth")
            self.assertTrue(best_state_path.is_file())
            self.assertFalse(Path(tmpdir, "latest.pth").exists())
            best_state = CheckpointManager.load(str(best_state_path))
            self.assertEqual(float(best_state["best_miou"]), 0.7)

            log_lines = Path(tmpdir, "val.log").read_text(encoding="utf-8").splitlines()
            self.assertTrue(
                any(
                    "Total accuracy: 88.0000" in line
                    for line in log_lines
                )
            )
            self.assertIn("  VALIDATION EPOCH 1", log_lines)
            self.assertIn("Validation Report", log_lines)
            self.assertTrue(any("Mean MIoU: 0.7000" in line for line in log_lines))
            self.assertIn("[MIoU_best: 0.7000]", log_lines)


if __name__ == "__main__":
    unittest.main()
