from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import override

from .logger import Logger


class TestNetLogger(Logger):
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
        message = (
            f"Train (epoch {epoch}/{max_epochs}) [{step}/{total_steps}]\t"
            f"Loss: {loss:.6f}\tAccuracy: {accuracy:.4f}\t"
            f"Time: {self.format_time(interval_time_seconds)} "
            f"(Elapsed: {self.format_time(epoch_elapsed_seconds)})"
        )
        return message

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
        summary = "Train summary: " + " | ".join(metric_parts)
        return summary

    @override
    def _format_validation_summary(self, val_metrics: dict[str, Any]) -> str | None:
        if not val_metrics:
            return None

        lines = [
            "Validation Report",
        ]
        if "pixels_processed" in val_metrics:
            lines.append(f"  Pixels processed: {int(val_metrics['pixels_processed'])}")
        if "accuracy" in val_metrics:
            lines.append(f"  Total accuracy: {float(val_metrics['accuracy']):.4f}")
        if "F1Score" in val_metrics:
            lines.append(f"  Mean F1Score: {float(val_metrics['F1Score']):.4f}")
        if "kappa" in val_metrics:
            lines.append(f"  Kappa: {float(val_metrics['kappa']):.4f}")
        if "MIoU" in val_metrics:
            lines.append(f"  Mean MIoU: {float(val_metrics['MIoU']):.4f}")

        if "confusion_matrix" in val_metrics:
            lines.extend(
                [
                    "",
                    "[Confusion matrix]",
                    self._format_matrix(val_metrics["confusion_matrix"]),
                ]
            )

        per_class_lines = self._format_per_class_metrics(val_metrics)
        if per_class_lines:
            lines.extend(["", "[Per-class metrics]", *per_class_lines])

        return "\n".join(lines)

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

    @staticmethod
    def _format_matrix(matrix: Any) -> str:
        rendered = np.array2string(np.asarray(matrix), max_line_width=120)
        return "\n".join(f"  {line}" for line in rendered.splitlines())

    @staticmethod
    def _format_metric(value: Any) -> str:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return "   nan"
        if np.isnan(scalar):
            return "   nan"
        return f"{scalar:7.4f}"

    def _format_per_class_metrics(self, val_metrics: dict[str, Any]) -> list[str]:
        required_keys = ["class_names", "per_class_accuracy", "per_class_f1", "per_class_iou"]
        if any(key not in val_metrics for key in required_keys):
            return []

        class_names = list(val_metrics["class_names"])
        per_class_accuracy = np.asarray(val_metrics["per_class_accuracy"])
        per_class_f1 = np.asarray(val_metrics["per_class_f1"])
        per_class_iou = np.asarray(val_metrics["per_class_iou"])
        class_width = max([12, *(len(str(name)) for name in class_names)])

        lines = [f"  {'class':<{class_width}} {'Acc':>7} {'F1':>7} {'IoU':>7}"]
        for index, class_name in enumerate(class_names):
            if index >= len(per_class_accuracy) or index >= len(per_class_f1) or index >= len(per_class_iou):
                break
            lines.append(
                f"  {str(class_name):<{class_width}} "
                f"{self._format_metric(per_class_accuracy[index])} "
                f"{self._format_metric(per_class_f1[index])} "
                f"{self._format_metric(per_class_iou[index])}"
            )
        return lines
