from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from engine import Trainer
from utils import CheckpointManager, MFNetLogger


class ScalarDataset(Dataset):
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        value = float(self.values[index])
        return {
            "x": torch.tensor([[value]], dtype=torch.float32),
            "y": torch.tensor([[2.0 * value]], dtype=torch.float32),
        }


class RegressionTrainer(Trainer):
    def train_forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        x = batch["x"].to(self.device)
        y = batch["y"].to(self.device)
        pred = self.model(x)
        loss = torch.mean((pred - y) ** 2)
        return loss, {"loss": float(loss.detach())}


class FixedLossTrainer(Trainer):
    def __init__(self, fixed_losses: list[float], *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.fixed_losses = fixed_losses
        self.loss_index = 0

    def train_forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del batch
        fixed_loss = float(self.fixed_losses[self.loss_index])
        self.loss_index += 1
        loss = self.model.weight.sum() * 0.0 + fixed_loss
        return loss, {"loss": fixed_loss}


class BeforeEpochTrainer(RegressionTrainer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.before_epoch_calls = 0

    def before_epoch(self) -> None:
        self.before_epoch_calls += 1


class IdentityInferencer:
    def run_batch_infer(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        return {
            "pred": model(batch["x"].to(device)),
            "target": batch["y"].to(device),
        }


class CaptureEvaluator:
    def __init__(self) -> None:
        self.last_outputs: list[dict[str, torch.Tensor | float]] | None = None

    def evaluate(
        self, outputs: list[dict[str, torch.Tensor | float]], **kwargs: object
    ) -> dict[str, float]:
        self.last_outputs = outputs
        return {"num_outputs": float(len(outputs))}


class DummyScheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1

    def state_dict(self) -> dict[str, int]:
        return {"step_calls": self.step_calls}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.step_calls = int(state_dict["step_calls"])


class CaptureStepTimeLogger(MFNetLogger):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.interval_times: list[float] = []

    def log_train_step(
        self,
        epoch: int,
        max_epochs: int,
        step: int,
        total_steps: int,
        step_stats: dict[str, float],
        interval_time_seconds: float,
        epoch_elapsed_seconds: float,
        global_step: int | None = None,
        lr: float | None = None,
    ) -> None:
        self.interval_times.append(interval_time_seconds)
        super().log_train_step(
            epoch=epoch,
            max_epochs=max_epochs,
            step=step,
            total_steps=total_steps,
            step_stats=step_stats,
            interval_time_seconds=interval_time_seconds,
            epoch_elapsed_seconds=epoch_elapsed_seconds,
            global_step=global_step,
            lr=lr,
        )


def build_trainer(
    work_dir: str,
    values: list[float],
    batch_size: int,
    effective_batch_size: int | None,
    log_step_interval: int = 1,
    max_epochs: int = 1,
    scheduler: DummyScheduler | None = None,
    save_step_interval: int = 0,
    evaluator: object | None = None,
    inferencer: object | None = None,
    trainer_cls: type[Trainer] = RegressionTrainer,
    logger_cls: type[MFNetLogger] = MFNetLogger,
) -> Trainer:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_loader = DataLoader(ScalarDataset(values), batch_size=batch_size, shuffle=False)

    return trainer_cls(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=[],
        logger=logger_cls(work_dir),
        checkpoint_manager=CheckpointManager(work_dir),
        evaluator=evaluator,
        device=torch.device("cpu"),
        scheduler=scheduler,
        inferencer=inferencer,
        cfg={
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "log_step_interval": log_step_interval,
            "val_epoch_interval": 100,
            "save_epoch_interval": 0,
            "save_step_interval": save_step_interval,
            **(
                {}
                if effective_batch_size is None
                else {"effective_batch_size": effective_batch_size}
            ),
        },
    )


class TrainerGradAccumTest(unittest.TestCase):
    def test_default_step_matches_micro_batch_when_accum_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0],
                batch_size=2,
                effective_batch_size=None,
            )

            train_metrics = trainer.train_one_epoch()

            self.assertEqual(trainer.grad_accum_steps, 1)
            self.assertEqual(trainer.global_step, 2)
            self.assertEqual(trainer.lr, trainer.optimizer.param_groups[0]["lr"])
            self.assertIn("loss", train_metrics)

            log_lines = Path(tmpdir, "train.log").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(log_lines), 2)
            self.assertIn("Train (epoch 1/1) [1/2]", log_lines[0])
            self.assertIn("Loss:", log_lines[0])
            self.assertIn("Accuracy:", log_lines[0])
            self.assertIn("Train (epoch 1/1) [2/2]", log_lines[1])

    def test_effective_batch_size_defaults_to_batch_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0],
                batch_size=2,
                effective_batch_size=None,
            )

            self.assertEqual(trainer.train_loader.batch_size, 2)
            self.assertEqual(trainer.grad_accum_steps, 1)

    def test_logged_interval_time_tracks_each_log_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0],
                batch_size=2,
                effective_batch_size=2,
                log_step_interval=2,
                logger_cls=CaptureStepTimeLogger,
            )

            trainer.train_one_epoch()

            assert isinstance(trainer.logger, CaptureStepTimeLogger)
            self.assertEqual(len(trainer.logger.interval_times), 1)
            self.assertGreaterEqual(trainer.logger.interval_times[0], 0.0)
            self.assertTrue(trainer.timer.has("epoch"))
            self.assertTrue(trainer.timer.has("log_interval"))
            self.assertGreaterEqual(trainer.timer.elapsed("epoch"), 0.0)

    def test_step_and_global_step_update_only_on_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = DummyScheduler()
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0],
                batch_size=2,
                effective_batch_size=4,
                scheduler=scheduler,
            )

            trainer.train_one_epoch()

            self.assertEqual(trainer.grad_accum_steps, 2)
            self.assertEqual(trainer.global_step, 1)
            self.assertEqual(scheduler.step_calls, 0)

            log_lines = Path(tmpdir, "train.log").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(log_lines), 1)
            self.assertIn("Train (epoch 1/1) [1/1]", log_lines[0])

    def test_tail_batches_are_flushed_as_last_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0, 5.0],
                batch_size=2,
                effective_batch_size=4,
            )

            trainer.train_one_epoch()

            self.assertEqual(trainer.global_step, 2)
            log_lines = Path(tmpdir, "train.log").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(log_lines), 2)
            self.assertIn("Train (epoch 1/1) [1/2]", log_lines[0])
            self.assertIn("Train (epoch 1/1) [2/2]", log_lines[1])

    def test_last_step_is_logged_when_interval_exceeds_epoch_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0],
                batch_size=1,
                effective_batch_size=1,
                log_step_interval=3,
            )

            trainer.train_one_epoch()

            log_lines = Path(tmpdir, "train.log").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(log_lines), 2)
            self.assertIn("Train (epoch 1/1) [3/4]", log_lines[0])
            self.assertIn("Train (epoch 1/1) [4/4]", log_lines[1])

    def test_last_step_is_not_logged_twice_when_it_matches_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0],
                batch_size=1,
                effective_batch_size=1,
                log_step_interval=2,
            )

            trainer.train_one_epoch()

            log_lines = Path(tmpdir, "train.log").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(log_lines), 2)
            self.assertIn("Train (epoch 1/1) [2/4]", log_lines[0])
            self.assertIn("Train (epoch 1/1) [4/4]", log_lines[1])

    def test_tail_flush_last_step_is_logged_even_when_interval_is_not_met(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0, 5.0],
                batch_size=2,
                effective_batch_size=4,
                log_step_interval=3,
            )

            trainer.train_one_epoch()

            log_lines = Path(tmpdir, "train.log").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(log_lines), 1)
            self.assertIn("Train (epoch 1/1) [2/2]", log_lines[0])

    def test_invalid_effective_batch_size_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "greater than or equal"):
                build_trainer(
                    work_dir=tmpdir,
                    values=[1.0, 2.0],
                    batch_size=2,
                    effective_batch_size=1,
                )

            with self.assertRaisesRegex(ValueError, "divisible"):
                build_trainer(
                    work_dir=tmpdir,
                    values=[1.0, 2.0, 3.0, 4.0],
                    batch_size=2,
                    effective_batch_size=3,
                )

    def test_checkpoint_resume_restores_optimizer_step_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0],
                batch_size=2,
                effective_batch_size=4,
                save_step_interval=1,
            )

            trainer.train_one_epoch()

            resumed = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0],
                batch_size=2,
                effective_batch_size=4,
            )
            state_dict = resumed.checkpoint_manager.load(
                path=str(Path(tmpdir) / "latest.pth")
            )
            resumed.model.load_state_dict(state_dict["model"])
            resumed.optimizer.load_state_dict(state_dict["optimizer"])
            if resumed.scheduler is not None and state_dict["scheduler"] is not None:
                resumed.scheduler.load_state_dict(state_dict["scheduler"])
            resumed.epoch = int(state_dict["epoch"])
            resumed.global_step = int(state_dict["global_step"])

            self.assertEqual(resumed.global_step, 1)

    def test_functional_train_step_forward_interface_runs_train_one_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0],
                batch_size=2,
                effective_batch_size=2,
            )

            train_metrics = trainer.train_one_epoch()

            self.assertIn("loss", train_metrics)
            self.assertEqual(trainer.global_step, 2)

    def test_epoch_metrics_average_over_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = torch.nn.Linear(1, 1, bias=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            train_loader = DataLoader(ScalarDataset([1.0, 2.0, 3.0]), batch_size=1, shuffle=False)
            trainer = FixedLossTrainer(
                fixed_losses=[1.0, 3.0, 100.0],
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=[],
                logger=MFNetLogger(tmpdir),
                checkpoint_manager=CheckpointManager(tmpdir),
                evaluator=CaptureEvaluator(),
                device=torch.device("cpu"),
                inferencer=IdentityInferencer(),
                cfg={
                    "max_epochs": 1,
                    "batch_size": 1,
                    "effective_batch_size": 1,
                    "log_step_interval": 1,
                    "val_epoch_interval": 100,
                    "save_epoch_interval": 0,
                    "save_step_interval": 0,
                },
            )

            train_metrics = trainer.train_one_epoch()

            self.assertAlmostEqual(train_metrics["loss"], (1.0 + 3.0 + 100.0) / 3.0)

    def test_validate_uses_raw_infer_outputs_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = CaptureEvaluator()
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0],
                batch_size=1,
                effective_batch_size=1,
                evaluator=evaluator,
                inferencer=IdentityInferencer(),
            )
            trainer.val_loader = DataLoader(ScalarDataset([1.0, 2.0]), batch_size=1, shuffle=False)

            trainer.validate()

            assert evaluator.last_outputs is not None
            self.assertEqual(len(evaluator.last_outputs), 2)
            self.assertIn("pred", evaluator.last_outputs[0])
            self.assertIn("target", evaluator.last_outputs[0])

    def test_train_raises_when_validation_enabled_without_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0],
                batch_size=1,
                effective_batch_size=1,
                evaluator=None,
                inferencer=None,
            )
            trainer.cfg["val_epoch_interval"] = 1
            trainer.max_epochs = 1

            with self.assertRaises(AttributeError):
                trainer.train()

    def test_train_runs_validation_when_enabled_and_dependencies_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = CaptureEvaluator()
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0],
                batch_size=1,
                effective_batch_size=1,
                evaluator=evaluator,
                inferencer=IdentityInferencer(),
            )
            trainer.val_loader = DataLoader(ScalarDataset([1.0, 2.0]), batch_size=1, shuffle=False)
            trainer.cfg["val_epoch_interval"] = 1
            trainer.max_epochs = 1

            trainer.train()

            assert evaluator.last_outputs is not None
            self.assertEqual(len(evaluator.last_outputs), 2)

    def test_before_epoch_hook_runs_once_per_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0],
                batch_size=1,
                effective_batch_size=1,
                trainer_cls=BeforeEpochTrainer,
            )
            trainer.max_epochs = 2
            trainer.cfg["save_epoch_interval"] = 100

            trainer.train()

            assert isinstance(trainer, BeforeEpochTrainer)
            self.assertEqual(trainer.before_epoch_calls, 2)

    def test_epoch_log_is_still_emitted_once_per_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0],
                batch_size=1,
                effective_batch_size=1,
                log_step_interval=3,
                max_epochs=1,
            )
            trainer.cfg["save_epoch_interval"] = 100

            trainer.train()

            log_lines = Path(tmpdir, "train.log").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(log_lines), 8)
            self.assertEqual(log_lines[0], "=" * 80)
            self.assertEqual(log_lines[1], "  EPOCH  1 / 1")
            self.assertEqual(log_lines[2], "=" * 80)
            self.assertIn("Train (epoch 1/1) [3/4]", log_lines[3])
            self.assertIn("Train (epoch 1/1) [4/4]", log_lines[4])
            self.assertTrue(log_lines[5].startswith("Training time: "))
            self.assertTrue(log_lines[6].startswith("Train summary: "))
            self.assertTrue(log_lines[7].startswith("Cumulative time: "))


if __name__ == "__main__":
    unittest.main()
