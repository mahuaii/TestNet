from __future__ import annotations

from typing_extensions import override

from .logger import Logger


class MFNetLogger(Logger):
    def __init__(
        self,
        work_dir: str,
        use_tensorboard: bool = True,
    ) -> None:
        super().__init__(
            work_dir=work_dir,
            use_tensorboard=use_tensorboard,
        )
        self._recent_losses: list[float] = []

    @override
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
        del global_step, lr
        loss = float(step_stats.get("loss", 0.0))
        accuracy = float(step_stats.get("accuracy", 0.0))
        return (
            f"[Train] (epoch {epoch}/{max_epochs}) [step {step}/{total_steps}]\t"
            f"Loss: {loss:.6f}\tAccuracy: {accuracy:.4f}\t"
            f"Interval: {self.format_time(interval_time_seconds)} "
            f"(Elapsed: {self.format_time(epoch_elapsed_seconds)})"
        )

    @override
    def _format_train_summary(
        self,
        train_metrics: dict[str, float],
        lr: float | None = None,
    ) -> str | None:
        metric_parts: list[str] = []
        if "loss" in train_metrics:
            metric_parts.append(f"Loss: {float(train_metrics['loss']):.6f}")
        if "accuracy" in train_metrics:
            metric_parts.append(f"Accuracy: {float(train_metrics['accuracy']):.4f}")
        if lr is not None:
            metric_parts.append(f"LR: {float(lr):.6f}")
        if not metric_parts:
            return None
        return "Train summary: " + " | ".join(metric_parts)

    @override
    def _format_validation_summary(self, val_metrics: dict[str, float]) -> str | None:
        preferred_order = ["MIoU", "accuracy", "F1Score", "kappa"]
        rendered: list[str] = []
        for key in preferred_order:
            if key in val_metrics:
                rendered.append(f"{key}: {float(val_metrics[key]):.4f}")
        for key, value in val_metrics.items():
            if key not in preferred_order:
                rendered.append(f"{key}: {float(value):.4f}")
        if not rendered:
            return None
        return "Validation metrics: " + " | ".join(rendered)

    @override
    def _write_step_scalars(
        self,
        global_step: int | None,
        step_stats: dict[str, float],
        lr: float | None = None,
    ) -> None:
        if global_step is None or "loss" not in step_stats:
            return None
        loss = float(step_stats["loss"])
        self._recent_losses.append(loss)
        smooth_loss = sum(self._recent_losses[-100:]) / min(len(self._recent_losses), 100)
        self._summary_writer.add_scalar("Loss/train", loss, global_step)
        self._summary_writer.add_scalar("Loss/train_smooth", smooth_loss, global_step)
        if lr is not None:
            self._summary_writer.add_scalar(
                "Learning_rate",
                float(lr),
                global_step,
            )
        self._summary_writer.flush()

    @override
    def _write_validation_scalars(self, epoch: int, val_metrics: dict[str, float]) -> None:
        metric_tags = {
            "MIoU": "Metrics/MIoU",
            "accuracy": "Metrics/Accuracy",
            "F1Score": "Metrics/F1Score",
            "kappa": "Metrics/Kappa",
        }
        for key, tag in metric_tags.items():
            if key in val_metrics:
                self._summary_writer.add_scalar(tag, float(val_metrics[key]), epoch)
        self._summary_writer.flush()
