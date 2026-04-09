from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class _NoOpSummaryWriter:
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        del tag, scalar_value, global_step

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class Logger(ABC):
    def __init__(
        self,
        work_dir: str,
        use_tensorboard: bool = True,
    ) -> None:
        self.log_path = Path(work_dir) / "train.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.use_tensorboard = bool(use_tensorboard)
        self._summary_writer = self._build_summary_writer(
            log_dir=str(Path(work_dir)),
        )

    def __enter__(self) -> Logger:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def log_epoch_start(self, epoch: int, max_epochs: int) -> None:
        self.log_message("")
        self.log_message("=" * 80)
        self.log_message(f"  EPOCH  {epoch} / {max_epochs}")
        self.log_message("=" * 80)

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
        message = self._format_train_step_message(
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
        if message is not None:
            self.log_message(message)
        self._write_step_scalars(
            global_step=global_step,
            step_stats=step_stats,
            lr=lr,
        )

    def log_epoch_end(
        self,
        train_time_seconds: float,
        train_metrics: dict[str, float],
        lr: float | None = None,
    ) -> None:
        self.log_message(f"Training time: {self.format_time(train_time_seconds)}")
        message = self._format_train_summary(
            train_metrics=train_metrics,
            lr=lr,
        )
        if message is not None:
            self.log_message(message)

    def log_validation_timing(
        self,
        test_time_seconds: float,
        epoch: int,
        val_metrics: dict[str, float] | None = None,
    ) -> None:
        self.log_message(f"Test time: {self.format_time(test_time_seconds)}")
        if val_metrics:
            message = self._format_validation_summary(val_metrics=val_metrics)
            if message is not None:
                self.log_message(message)
            self._write_validation_scalars(epoch=epoch, val_metrics=val_metrics)

    def log_checkpoint_saved(self, path: str | Path) -> None:
        self.log_message(f"Saved checkpoint: {path}", False)

    def log_best_metric(self, name: str, value: float) -> None:
        self.log_message(f"{name}: {value:.4f}")

    def log_message(self, message: str, writefile: bool = True) -> None:
        print(message)
        if writefile:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(message + "\n")

    def close(self) -> None:
        self._summary_writer.close()

    @abstractmethod
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
    ) -> str | None: ...

    @abstractmethod
    def _format_train_summary(
        self,
        train_metrics: dict[str, float],
        lr: float | None = None,
    ) -> str | None: ...

    @abstractmethod
    def _format_validation_summary(self, val_metrics: dict[str, float]) -> str | None: ...

    @abstractmethod
    def _write_step_scalars(
        self,
        global_step: int | None,
        step_stats: dict[str, float],
        lr: float | None = None,
    ) -> None: ...

    @abstractmethod
    def _write_validation_scalars(self, epoch: int, val_metrics: dict[str, float]) -> None: ...

    @staticmethod
    def format_time(seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours}:{minutes:02d}:{secs:02d}"

    def _build_summary_writer(
        self,
        log_dir: str,
    ) -> Any:
        if not self.use_tensorboard:
            return _NoOpSummaryWriter()
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except Exception:
            return _NoOpSummaryWriter()
        return SummaryWriter(log_dir)
