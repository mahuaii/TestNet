from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from engine import GradAccumTrainer, MFNetDGATrainer, Trainer
from utils import CheckpointManager, TestNetLogger, StatTracker


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


class GradAccumRegressionTrainer(GradAccumTrainer):
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


class GradAccumFixedLossTrainer(GradAccumTrainer):
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


class FakeDGABlock(torch.nn.Module):
    def __init__(self, alpha: float, beta: float) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha))
        self.beta = torch.nn.Parameter(torch.tensor(beta))


class FakeDGAModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.dga_blocks = torch.nn.ModuleList(
            [
                FakeDGABlock(alpha=0.1, beta=0.2),
                FakeDGABlock(alpha=0.3, beta=0.4),
            ]
        )


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


class SequenceMetricsEvaluator:
    def __init__(self, metrics_sequence: list[dict[str, float]]) -> None:
        self.metrics_sequence = metrics_sequence
        self.call_index = 0

    def evaluate(
        self, outputs: list[dict[str, torch.Tensor | float]], **kwargs: object
    ) -> dict[str, float]:
        del outputs, kwargs
        metrics = self.metrics_sequence[self.call_index]
        self.call_index += 1
        return metrics


class DummyScheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1

    def state_dict(self) -> dict[str, int]:
        return {"step_calls": self.step_calls}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.step_calls = int(state_dict["step_calls"])


class CaptureStepTimeLogger(TestNetLogger):
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


class CaptureStepStatsLogger(TestNetLogger):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.step_stats_history: list[dict[str, float]] = []

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
        self.step_stats_history.append(dict(step_stats))
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
    logger_cls: type[TestNetLogger] = TestNetLogger,
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
        evaluator=evaluator,
        device=torch.device("cpu"),
        scheduler=scheduler,
        inferencer=inferencer,
        cfg={
            "work_dir": work_dir,
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

            self.assertEqual(trainer.global_step, 2)
            self.assertEqual(trainer.lr, trainer.optimizer.param_groups[0]["lr"])
            self.assertIn("loss", train_metrics)

            log_lines = Path(tmpdir, "train.log").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(log_lines), 2)
            self.assertIn("Train (epoch 1/1) [1/2]", log_lines[0])
            self.assertIn("Loss:", log_lines[0])
            self.assertIn("Accuracy:", log_lines[0])
            self.assertIn("Train (epoch 1/1) [2/2]", log_lines[1])

    def test_base_trainer_ignores_effective_batch_size_and_hides_grad_accum_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0],
                batch_size=2,
                effective_batch_size=4,
            )

            train_metrics = trainer.train_one_epoch()

            self.assertEqual(trainer.train_loader.batch_size, 2)
            self.assertEqual(trainer.total_steps_per_epoch, 2)
            self.assertEqual(trainer.global_step, 2)
            self.assertFalse(hasattr(trainer, "grad_accum_steps"))
            self.assertIn("loss", train_metrics)

    def test_grad_accum_trainer_defaults_effective_batch_size_to_batch_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0],
                batch_size=2,
                effective_batch_size=None,
                trainer_cls=GradAccumRegressionTrainer,
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

    def test_base_trainer_logs_window_averaged_step_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = torch.nn.Linear(1, 1, bias=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            train_loader = DataLoader(ScalarDataset([1.0, 2.0, 3.0]), batch_size=1, shuffle=False)
            trainer = FixedLossTrainer(
                fixed_losses=[1.0, 3.0, 9.0],
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=[],
                logger=CaptureStepStatsLogger(tmpdir),
                evaluator=CaptureEvaluator(),
                device=torch.device("cpu"),
                inferencer=IdentityInferencer(),
                cfg={
                    "work_dir": tmpdir,
                    "max_epochs": 1,
                    "batch_size": 1,
                    "log_step_interval": 2,
                    "val_epoch_interval": 100,
                    "save_epoch_interval": 0,
                    "save_step_interval": 0,
                },
            )

            trainer.train_one_epoch()

            assert isinstance(trainer.logger, CaptureStepStatsLogger)
            self.assertEqual(len(trainer.logger.step_stats_history), 2)
            self.assertAlmostEqual(trainer.logger.step_stats_history[0]["loss"], 2.0)
            self.assertAlmostEqual(trainer.logger.step_stats_history[1]["loss"], 9.0)

    def test_mfnet_dga_trainer_logs_each_block_alpha_beta_point_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = FakeDGAModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            train_loader = DataLoader(ScalarDataset([1.0]), batch_size=1, shuffle=False)
            trainer = MFNetDGATrainer(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=[],
                logger=CaptureStepStatsLogger(tmpdir),
                evaluator=CaptureEvaluator(),
                device=torch.device("cpu"),
                inferencer=IdentityInferencer(),
                cfg={
                    "work_dir": tmpdir,
                    "max_epochs": 1,
                    "batch_size": 1,
                    "log_step_interval": 1,
                    "val_epoch_interval": 100,
                    "save_epoch_interval": 0,
                    "save_step_interval": 0,
                },
            )
            tracker = StatTracker()
            tracker.update_mean_stats({"loss": 2.0})
            trainer.timer.mark("epoch")
            trainer.timer.mark("log_interval")

            trainer.after_step(step=1, step_stats_tracker=tracker)

            assert isinstance(trainer.logger, CaptureStepStatsLogger)
            self.assertEqual(len(trainer.logger.step_stats_history), 1)
            step_stats = trainer.logger.step_stats_history[0]
            self.assertAlmostEqual(step_stats["dga/alpha_block_0"], 0.1, places=6)
            self.assertAlmostEqual(step_stats["dga/beta_block_0"], 0.2, places=6)
            self.assertAlmostEqual(step_stats["dga/alpha_block_1"], 0.3, places=6)
            self.assertAlmostEqual(step_stats["dga/beta_block_1"], 0.4, places=6)
            self.assertNotIn("dga/alpha_mean", step_stats)
            self.assertNotIn("dga/beta_mean", step_stats)

    def test_grad_accum_trainer_logs_step_interval_window_average(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = torch.nn.Linear(1, 1, bias=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            train_loader = DataLoader(ScalarDataset([1.0, 2.0, 3.0, 4.0]), batch_size=1, shuffle=False)
            trainer = GradAccumFixedLossTrainer(
                fixed_losses=[1.0, 3.0, 5.0, 7.0],
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=[],
                logger=CaptureStepStatsLogger(tmpdir),
                evaluator=CaptureEvaluator(),
                device=torch.device("cpu"),
                inferencer=IdentityInferencer(),
                cfg={
                    "work_dir": tmpdir,
                    "max_epochs": 1,
                    "batch_size": 1,
                    "effective_batch_size": 2,
                    "log_step_interval": 2,
                    "val_epoch_interval": 100,
                    "save_epoch_interval": 0,
                    "save_step_interval": 0,
                },
            )

            trainer.train_one_epoch()

            assert isinstance(trainer.logger, CaptureStepStatsLogger)
            self.assertEqual(len(trainer.logger.step_stats_history), 1)
            self.assertAlmostEqual(trainer.logger.step_stats_history[0]["loss"], 4.0)

    def test_step_and_global_step_update_only_on_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = DummyScheduler()
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0],
                batch_size=2,
                effective_batch_size=4,
                scheduler=scheduler,
                trainer_cls=GradAccumRegressionTrainer,
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
                trainer_cls=GradAccumRegressionTrainer,
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
                trainer_cls=GradAccumRegressionTrainer,
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
                    trainer_cls=GradAccumRegressionTrainer,
                )

            with self.assertRaisesRegex(ValueError, "divisible"):
                build_trainer(
                    work_dir=tmpdir,
                    values=[1.0, 2.0, 3.0, 4.0],
                    batch_size=2,
                    effective_batch_size=3,
                    trainer_cls=GradAccumRegressionTrainer,
                )

    def test_checkpoint_resume_restores_recorded_epoch_and_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0],
                batch_size=2,
                effective_batch_size=4,
                save_step_interval=1,
                trainer_cls=GradAccumRegressionTrainer,
            )

            trainer.train()

            resumed = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0, 3.0, 4.0],
                batch_size=2,
                effective_batch_size=4,
                trainer_cls=GradAccumRegressionTrainer,
            )
            latest_path = Path(tmpdir) / "latest.pth"
            state_dict = CheckpointManager.load(path=str(latest_path))
            self.assertEqual(state_dict["epoch"], 1)

            resumed._load_training_state(str(latest_path))

            self.assertEqual(resumed.epoch, 1)
            self.assertEqual(resumed.global_step, 1)

    def test_checkpoint_persists_best_miou_in_named_and_latest_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0],
                batch_size=1,
                effective_batch_size=1,
                save_step_interval=1,
            )
            trainer.best_miou = 0.7

            trainer.train()

            named_state = CheckpointManager.load(str(Path(tmpdir) / "global_step_1.pth"))
            latest_state = CheckpointManager.load(str(Path(tmpdir) / "latest.pth"))
            self.assertEqual(float(named_state["best_miou"]), 0.7)
            self.assertEqual(float(latest_state["best_miou"]), 0.7)

    def test_load_training_state_defaults_best_miou_for_legacy_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0],
                batch_size=1,
                effective_batch_size=1,
            )
            legacy_path = Path(tmpdir) / "legacy.pth"
            torch.save(
                {
                    "model": trainer.model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "scheduler": None,
                    "epoch": 3,
                    "global_step": 5,
                },
                legacy_path,
            )

            trainer._load_training_state(str(legacy_path))

            self.assertEqual(trainer.best_miou, 0.0)

    def test_load_training_state_restores_best_miou(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0],
                batch_size=1,
                effective_batch_size=1,
            )
            state_path = Path(tmpdir) / "resume.pth"
            torch.save(
                {
                    "model": trainer.model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "scheduler": None,
                    "epoch": 4,
                    "global_step": 7,
                    "best_miou": 0.82,
                },
                state_path,
            )

            trainer._load_training_state(str(state_path))

            self.assertEqual(trainer.epoch, 4)
            self.assertEqual(trainer.global_step, 7)
            self.assertEqual(trainer.best_miou, 0.82)

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
                logger=TestNetLogger(tmpdir),
                evaluator=CaptureEvaluator(),
                device=torch.device("cpu"),
                inferencer=IdentityInferencer(),
                cfg={
                    "work_dir": tmpdir,
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

    def test_validate_updates_best_miou_and_logs_when_metric_improves(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = SequenceMetricsEvaluator([{"MIoU": 0.6, "accuracy": 88.0}])
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

            self.assertEqual(trainer.best_miou, 0.6)
            log_lines = Path(tmpdir, "train.log").read_text(encoding="utf-8").splitlines()
            self.assertIn("[MIoU_best: 0.6000]", log_lines[-1])

    def test_validate_keeps_best_miou_when_metric_does_not_improve(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = SequenceMetricsEvaluator([{"MIoU": 0.6}, {"MIoU": 0.5}])
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
            trainer.validate()

            self.assertEqual(trainer.best_miou, 0.6)
            log_lines = Path(tmpdir, "train.log").read_text(encoding="utf-8").splitlines()
            self.assertEqual(sum(line == "[MIoU_best: 0.6000]" for line in log_lines), 2)

    def test_validate_ignores_missing_miou_for_best_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = SequenceMetricsEvaluator([{"accuracy": 88.0}])
            trainer = build_trainer(
                work_dir=tmpdir,
                values=[1.0, 2.0],
                batch_size=1,
                effective_batch_size=1,
                evaluator=evaluator,
                inferencer=IdentityInferencer(),
            )
            trainer.best_miou = 0.4
            trainer.val_loader = DataLoader(ScalarDataset([1.0, 2.0]), batch_size=1, shuffle=False)

            trainer.validate()

            self.assertEqual(trainer.best_miou, 0.4)
            log_lines = Path(tmpdir, "train.log").read_text(encoding="utf-8").splitlines()
            self.assertIn("[MIoU_best: 0.4000]", log_lines)

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
            self.assertEqual(len(log_lines), 10)
            self.assertEqual(log_lines[0], "=" * 80)
            self.assertEqual(log_lines[1], "  EPOCH 1 / 1")
            self.assertEqual(log_lines[2], "=" * 80)
            self.assertIn("Train (epoch 1/1) [3/4]", log_lines[3])
            self.assertIn("Train (epoch 1/1) [4/4]", log_lines[4])
            self.assertEqual(log_lines[5], "-" * 80)
            self.assertEqual(log_lines[6], "  TRAINING SUMMARY")
            self.assertEqual(log_lines[7], "-" * 80)
            self.assertTrue(log_lines[8].startswith("Training time: "))
            self.assertTrue(log_lines[9].startswith("Train summary: "))


if __name__ == "__main__":
    unittest.main()
