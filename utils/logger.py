from __future__ import annotations

from pathlib import Path


class Logger:
    def __init__(self, work_dir: str) -> None:
        self.log_path = Path(work_dir) / "train.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_iter(
        self,
        epoch: int,
        iter_idx: int,
        global_step: int,
        running_metrics: dict[str, float],
        state_metrics: dict[str, float] | None = None,
    ) -> None:
        message = f"[train] epoch={epoch} iter={iter_idx} step={global_step}"
        message += self._format_metric_group("running", running_metrics)
        if state_metrics:
            message += self._format_metric_group("state", state_metrics)
        self.log_message(message)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None = None,
        train_state_metrics: dict[str, float] | None = None,
    ) -> None:
        message = f"[epoch] epoch={epoch}"
        message += self._format_metric_group("train", train_metrics)
        if train_state_metrics:
            message += self._format_metric_group("train_state", train_state_metrics)
        if val_metrics:
            message += self._format_metric_group("val", val_metrics)
        self.log_message(message)

    def log_message(self, message: str) -> None:
        print(message)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    def _format_metric_group(self, group_name: str, metrics: dict[str, float]) -> str:
        formatted_metrics = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        return f" {group_name}[{formatted_metrics}]"
