from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import Logger, MFNetDGALogger, MFNetLogger


class _FakeSummaryWriter:
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self.scalars: list[tuple[str, float, int]] = []
        self.flush_calls = 0
        self.closed = False

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.scalars.append((tag, float(scalar_value), int(global_step)))

    def flush(self) -> None:
        self.flush_calls += 1

    def close(self) -> None:
        self.closed = True


class _BaseStyleLogger(Logger):
    def _format_train_step_message(
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
    ) -> str | None:
        del epoch, max_epochs, step, total_steps, step_stats
        del interval_time_seconds, epoch_elapsed_seconds, global_step, lr
        return None

    def _format_train_summary(
        self,
        train_metrics: dict[str, float],
        lr: float | None = None,
    ) -> str | None:
        del train_metrics, lr
        return None

    def _format_validation_summary(self, val_metrics: dict[str, float]) -> str | None:
        del val_metrics
        return None

    def _write_step_scalars(
        self,
        global_step: int | None,
        step_stats: dict[str, float],
        lr: float | None = None,
    ) -> None:
        del global_step, step_stats, lr
        return None

    def _write_validation_scalars(self, epoch: int, val_metrics: dict[str, float]) -> None:
        del epoch, val_metrics
        return None


class _PurgeTrackingLogger(_BaseStyleLogger):
    def _build_summary_writer(
        self,
        log_dir: str,
        purge_step: int | None = None,
    ) -> _FakeSummaryWriter:
        self.build_calls = getattr(self, "build_calls", [])
        self.build_calls.append((log_dir, purge_step))
        return _FakeSummaryWriter(log_dir)


class LoggerTest(unittest.TestCase):
    def test_base_logger_only_emits_generic_timing_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _BaseStyleLogger(tmpdir, use_tensorboard=False)

            logger.log_epoch_start(epoch=1, max_epochs=3)
            logger.log_train_step(
                epoch=1,
                max_epochs=3,
                step=2,
                total_steps=5,
                step_stats={"loss": 1.25, "accuracy": 87.5},
                interval_time_seconds=65,
                epoch_elapsed_seconds=130,
                global_step=2,
                lr=0.01,
            )
            logger.log_epoch_end(
                train_time_seconds=3661,
                train_metrics={"loss": 0.5, "accuracy": 90.0},
                lr=0.01,
            )
            logger.log_validation_timing(
                test_time_seconds=61,
                epoch=2,
                val_metrics={"MIoU": 0.6, "accuracy": 88.0},
            )
            logger.close()

            log_lines = Path(tmpdir, "train.log").read_text(encoding="utf-8").splitlines()
            self.assertEqual(log_lines[0], "")
            self.assertEqual(log_lines[1], "=" * 80)
            self.assertEqual(log_lines[2], "  EPOCH 1 / 3")
            self.assertEqual(log_lines[3], "=" * 80)
            self.assertEqual(log_lines[4], "-" * 80)
            self.assertEqual(log_lines[5], "  TRAINING SUMMARY")
            self.assertEqual(log_lines[6], "-" * 80)
            self.assertEqual(log_lines[7], "Training time: 1:01:01")
            self.assertEqual(log_lines[8], "")
            self.assertEqual(log_lines[9], "-" * 80)
            self.assertEqual(log_lines[10], "  VALIDATION EPOCH 2")
            self.assertEqual(log_lines[11], "-" * 80)
            self.assertEqual(log_lines[12], "Validation time: 0:01:01")

    def test_mfnet_logger_formats_mfnet_style_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MFNetLogger(tmpdir, use_tensorboard=False)

            logger.log_epoch_start(epoch=1, max_epochs=3)
            logger.log_train_step(
                epoch=1,
                max_epochs=3,
                step=2,
                total_steps=5,
                step_stats={"loss": 1.25, "accuracy": 87.5},
                interval_time_seconds=65,
                epoch_elapsed_seconds=130,
                global_step=2,
                lr=0.01,
            )
            logger.log_epoch_end(
                train_time_seconds=3661,
                train_metrics={"loss": 0.5, "accuracy": 90.0},
                lr=0.01,
            )
            logger.log_validation_timing(
                test_time_seconds=61,
                epoch=2,
                val_metrics={"MIoU": 0.6, "accuracy": 88.0},
            )
            logger.log_checkpoint_saved("/tmp/model.pth")
            logger.log_best_metric("MIoU_best", 0.6)
            logger.close()

            log_lines = Path(tmpdir, "train.log").read_text(encoding="utf-8").splitlines()
            self.assertEqual(log_lines[0], "")
            self.assertEqual(log_lines[1], "=" * 80)
            self.assertEqual(log_lines[2], "  EPOCH 1 / 3")
            self.assertEqual(log_lines[3], "=" * 80)
            self.assertIn("Train (epoch 1/3) [2/5]", log_lines[4])
            self.assertIn("Loss: 1.250000", log_lines[4])
            self.assertIn("Accuracy: 87.5000", log_lines[4])
            self.assertIn("Time: 0:01:05", log_lines[4])
            self.assertIn("(Elapsed: 0:02:10)", log_lines[4])
            self.assertEqual(log_lines[5], "-" * 80)
            self.assertEqual(log_lines[6], "  TRAINING SUMMARY")
            self.assertEqual(log_lines[7], "-" * 80)
            self.assertEqual(log_lines[8], "Training time: 1:01:01")
            self.assertIn(
                "Train summary: Loss: 0.500000 | Accuracy: 90.0000 | LR: 0.010000",
                log_lines[9],
            )
            self.assertEqual(log_lines[10], "")
            self.assertEqual(log_lines[11], "-" * 80)
            self.assertEqual(log_lines[12], "  VALIDATION EPOCH 2")
            self.assertEqual(log_lines[13], "-" * 80)
            self.assertEqual(log_lines[14], "Validation time: 0:01:01")
            self.assertEqual(log_lines[15], "Validation Report")
            self.assertIn("Total accuracy: 88.0000", log_lines[16])
            self.assertIn("Mean MIoU: 0.6000", "\n".join(log_lines))
            self.assertEqual(log_lines[-1], "[MIoU_best: 0.6000]")

    def test_truncate_after_completed_epoch_removes_incomplete_epoch_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MFNetLogger(tmpdir, use_tensorboard=False)
            logger.log_message("Experiment: resume-test")
            logger.log_epoch_start(epoch=1, max_epochs=3)
            logger.log_epoch_end(
                train_time_seconds=10,
                train_metrics={"loss": 0.5, "accuracy": 90.0},
                lr=0.01,
            )
            logger.log_epoch_start(epoch=2, max_epochs=3)
            logger.log_train_step(
                epoch=2,
                max_epochs=3,
                step=1,
                total_steps=5,
                step_stats={"loss": 1.25, "accuracy": 87.5},
                interval_time_seconds=1,
                epoch_elapsed_seconds=1,
                global_step=6,
                lr=0.01,
            )

            changed = logger.truncate_after_completed_epoch(completed_epoch=1)

            self.assertTrue(changed)
            log_text = Path(tmpdir, "train.log").read_text(encoding="utf-8")
            self.assertIn("Experiment: resume-test", log_text)
            self.assertIn("  EPOCH 1 / 3", log_text)
            self.assertIn("Train summary:", log_text)
            self.assertNotIn("  EPOCH 2 / 3", log_text)
            self.assertNotIn("Train (epoch 2/3)", log_text)
            logger.close()

    def test_truncate_after_completed_epoch_is_noop_when_log_is_already_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MFNetLogger(tmpdir, use_tensorboard=False)
            logger.log_epoch_start(epoch=1, max_epochs=1)
            logger.log_epoch_end(
                train_time_seconds=10,
                train_metrics={"loss": 0.5},
            )

            changed = logger.truncate_after_completed_epoch(completed_epoch=1)

            self.assertFalse(changed)
            log_text = Path(tmpdir, "train.log").read_text(encoding="utf-8")
            self.assertIn("  EPOCH 1 / 1", log_text)
            logger.close()

    def test_purge_tensorboard_after_global_step_reopens_writer_with_next_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _PurgeTrackingLogger(tmpdir, use_tensorboard=True)
            original_writer = logger._summary_writer

            logger.purge_tensorboard_after_global_step(global_step=7)

            self.assertTrue(original_writer.closed)
            self.assertEqual(
                logger.build_calls,
                [(tmpdir, None), (tmpdir, 8)],
            )
            logger.close()

    def test_mfnet_logger_tensorboard_scalars_are_written_with_fake_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MFNetLogger(tmpdir, use_tensorboard=False)
            logger._summary_writer = _FakeSummaryWriter(tmpdir)

            logger.log_train_step(
                epoch=1,
                max_epochs=2,
                step=1,
                total_steps=4,
                step_stats={"loss": 2.0, "accuracy": 50.0},
                interval_time_seconds=1,
                epoch_elapsed_seconds=2,
                global_step=7,
                lr=0.01,
            )
            logger.log_validation_timing(
                test_time_seconds=1,
                epoch=1,
                val_metrics={"MIoU": 0.7, "F1Score": 0.8},
            )

            writer = logger._summary_writer
            assert isinstance(writer, _FakeSummaryWriter)
            self.assertIn(("Loss/train", 2.0, 7), writer.scalars)
            self.assertIn(("Loss/train_smooth", 2.0, 7), writer.scalars)
            self.assertIn(("Learning_rate", 0.01, 7), writer.scalars)
            self.assertIn(("Metrics/MIoU", 0.7, 1), writer.scalars)
            self.assertIn(("Metrics/F1Score", 0.8, 1), writer.scalars)
            logger.close()
            self.assertTrue(writer.closed)

    def test_mfnet_dga_logger_writes_dga_tensorboard_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MFNetDGALogger(tmpdir, use_tensorboard=False)
            logger._summary_writer = _FakeSummaryWriter(tmpdir)

            logger.log_train_step(
                epoch=1,
                max_epochs=2,
                step=1,
                total_steps=4,
                step_stats={
                    "loss": 2.0,
                    "accuracy": 50.0,
                    "dga/alpha_block_0": 0.1,
                    "dga/beta_block_0": 0.2,
                    "dga/alpha_block_1": 0.3,
                    "dga/beta_block_1": 0.4,
                },
                interval_time_seconds=1,
                epoch_elapsed_seconds=2,
                global_step=7,
                lr=0.01,
            )

            writer = logger._summary_writer
            assert isinstance(writer, _FakeSummaryWriter)
            self.assertIn(("Loss/train", 2.0, 7), writer.scalars)
            self.assertIn(("Learning_rate", 0.01, 7), writer.scalars)
            self.assertIn(("DGA/alpha_block_0", 0.1, 7), writer.scalars)
            self.assertIn(("DGA/beta_block_0", 0.2, 7), writer.scalars)
            self.assertIn(("DGA/alpha_block_1", 0.3, 7), writer.scalars)
            self.assertIn(("DGA/beta_block_1", 0.4, 7), writer.scalars)
            log_text = Path(tmpdir, "train.log").read_text(encoding="utf-8")
            self.assertNotIn("alpha_block", log_text)
            self.assertNotIn("beta_block", log_text)
            logger.close()

    def test_mfnet_logger_does_not_write_dga_tensorboard_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MFNetLogger(tmpdir, use_tensorboard=False)
            logger._summary_writer = _FakeSummaryWriter(tmpdir)

            logger.log_train_step(
                epoch=1,
                max_epochs=2,
                step=1,
                total_steps=4,
                step_stats={
                    "loss": 2.0,
                    "dga/alpha_block_0": 0.1,
                    "dga/beta_block_0": 0.2,
                },
                interval_time_seconds=1,
                epoch_elapsed_seconds=2,
                global_step=7,
                lr=0.01,
            )

            writer = logger._summary_writer
            assert isinstance(writer, _FakeSummaryWriter)
            self.assertIn(("Loss/train", 2.0, 7), writer.scalars)
            self.assertNotIn(("DGA/alpha_block_0", 0.1, 7), writer.scalars)
            self.assertNotIn(("DGA/beta_block_0", 0.2, 7), writer.scalars)
            logger.close()

    def test_base_logger_does_not_write_metric_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = _BaseStyleLogger(tmpdir, use_tensorboard=False)
            logger._summary_writer = _FakeSummaryWriter(tmpdir)
            logger.log_train_step(
                epoch=1,
                max_epochs=1,
                step=1,
                total_steps=1,
                step_stats={"loss": 1.0, "accuracy": 100.0},
                interval_time_seconds=0,
                epoch_elapsed_seconds=0,
                global_step=1,
            )
            logger.log_validation_timing(
                test_time_seconds=0,
                epoch=1,
                val_metrics={"MIoU": 0.5},
            )
            writer = logger._summary_writer
            assert isinstance(writer, _FakeSummaryWriter)
            self.assertEqual(writer.scalars, [])
            logger.close()

    def test_missing_tensorboard_dependency_falls_back_to_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("builtins.__import__", side_effect=ImportError):
                logger = MFNetLogger(tmpdir, use_tensorboard=True)
                logger.log_train_step(
                    epoch=1,
                    max_epochs=1,
                    step=1,
                    total_steps=1,
                    step_stats={"loss": 1.0, "accuracy": 100.0},
                    interval_time_seconds=0,
                    epoch_elapsed_seconds=0,
                    global_step=1,
                )
                logger.close()


if __name__ == "__main__":
    unittest.main()
