from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


# Empty SummaryWriter replacement used when TensorBoard is disabled or unavailable.
class _NoOpSummaryWriter:
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        del tag, scalar_value, global_step

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class Logger(ABC):
    _SECTION_WIDTH = 80
    _EPOCH_HEADER_PATTERN = re.compile(r"^\s*EPOCH\s+(\d+)\s*/")
    _DIAGNOSTIC_TENSORBOARD_PREFIXES = {
        "prealign/": "PreAlign/",
        "spmf/structure/": "SPMF/Structure/",
        "spmf/": "SPMF/",
    }

    def __init__(
        self,
        work_dir: str,
        use_tensorboard: bool = True,
    ) -> None:
        self.log_path = Path(work_dir) / "train.log"
        self.val_log_path = Path(work_dir) / "val.log"
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
        self._log_section_header(f"EPOCH {epoch} / {max_epochs}", fill="=")

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

    def log_epoch_end(
        self,
        train_time_seconds: float,
        train_metrics: dict[str, float],
        lr: float | None = None,
    ) -> None:
        self._log_section_header("TRAINING SUMMARY", fill="-")
        self.log_message(f"Training time: {self.format_time(train_time_seconds)}")
        message = self._format_train_summary(
            train_metrics=train_metrics,
            lr=lr,
        )
        if message is not None:
            self.log_message(message)

    def log_lr_groups(
        self,
        *,
        epoch: int,
        max_epochs: int,
        stage_label: str,
        scheduler_scale: float,
        group_summaries: Sequence[Mapping[str, Any]],
    ) -> None:
        self.log_message(
            f"LR groups | Epoch: {epoch}/{max_epochs} | Stage: {stage_label} | "
            f"Scale: {scheduler_scale:.8g}"
        )
        formatted_groups = [
            self._format_lr_group_summary(group)
            for group in group_summaries
        ]
        if not formatted_groups:
            self.log_message("  (no optimizer groups)")
            return

        scope_width = max(
            len("scope"),
            *(len(group["scope"]) for group in formatted_groups),
        )
        group_width = max(
            len("group"),
            *(len(group["group"]) for group in formatted_groups),
        )
        lr_width = max(
            len("lr"),
            *(len(group["lr"]) for group in formatted_groups),
        )
        params_width = max(
            len("params"),
            *(len(group["params"]) for group in formatted_groups),
        )
        self.log_message(
            f"  {'scope':<{scope_width}}  {'group':<{group_width}}  "
            f"{'lr':<{lr_width}}  {'params':>{params_width}}"
        )
        for formatted in formatted_groups:
            self.log_message(
                f"  {formatted['scope']:<{scope_width}}  "
                f"{formatted['group']:<{group_width}}  "
                f"{formatted['lr']:<{lr_width}}  "
                f"{formatted['params']:>{params_width}}"
            )

    def log_trainable_param_counts(
        self,
        *,
        epoch: int,
        stage_label: str,
        counts: Mapping[str, int],
    ) -> None:
        self._log_section_header("TRAINABLE PARAMS", fill="-")
        self.log_message(f"Epoch: {epoch} | Stage: {stage_label}")
        for name, count in counts.items():
            self.log_message(f"  {name}: {int(count)}")

    def log_diagnostic_summary(
        self,
        *,
        epoch: int,
        metrics: Mapping[str, float],
        prefixes: Sequence[str],
    ) -> None:
        selected = self._select_diagnostic_metrics(metrics=metrics, prefixes=prefixes)
        if not selected:
            return

        self._log_section_header("DIAGNOSTIC SUMMARY", fill="-")
        self.log_message(f"Epoch: {epoch}")
        for key, value in selected.items():
            self.log_message(f"  {key}: {value:.6g}")

    def write_module_norm_scalars(
        self,
        *,
        global_step: int,
        param_group_stats: Mapping[str, Mapping[str, float]],
        module_group_stats: Mapping[str, Mapping[str, float]],
    ) -> None:
        for group_name, metrics in param_group_stats.items():
            for metric_name, value in metrics.items():
                self._summary_writer.add_scalar(
                    f"ModuleNorm/ParamGroup/{group_name}/{metric_name}",
                    float(value),
                    global_step,
                )
        for group_name, metrics in module_group_stats.items():
            for metric_name, value in metrics.items():
                self._summary_writer.add_scalar(
                    f"ModuleNorm/Module/{group_name}/{metric_name}",
                    float(value),
                    global_step,
                )
        self._summary_writer.flush()

    def log_validation_timing(
        self,
        test_time_seconds: float,
        epoch: int,
        val_metrics: dict[str, float] | None = None,
    ) -> None:
        self.log_val_message("")
        self._log_val_section_header(f"VALIDATION EPOCH {epoch}", fill="-")
        self.log_val_message(f"Validation time: {self.format_time(test_time_seconds)}")
        if val_metrics:
            message = self._format_validation_summary(val_metrics=val_metrics)
            if message is not None:
                self.log_val_message(message)

    def log_checkpoint_saved(self, path: str | Path) -> None:
        self.log_message(f"Saved checkpoint: {path}", False)

    def log_best_metric(self, name: str, value: float) -> None:
        self.log_message(f"[{name}: {value:.4f}]")

    def log_val_best_metric(self, name: str, value: float) -> None:
        self.log_val_message(f"[{name}: {value:.4f}]")

    def log_message(self, message: str, writefile: bool = True) -> None:
        self._log_message_to_path(self.log_path, message, writefile=writefile)

    def log_val_message(self, message: str, writefile: bool = True) -> None:
        self._log_message_to_path(self.val_log_path, message, writefile=writefile)

    @staticmethod
    def _log_message_to_path(path: Path, message: str, writefile: bool = True) -> None:
        print(message)
        if writefile:
            with path.open("a", encoding="utf-8") as f:
                f.write(message + "\n")

    def truncate_after_completed_epoch(self, completed_epoch: int) -> bool:
        if not self.log_path.is_file():
            return False

        lines = self.log_path.read_text(encoding="utf-8").splitlines(keepends=True)
        truncate_at: int | None = None
        for index, line in enumerate(lines):
            match = self._EPOCH_HEADER_PATTERN.match(line)
            if match is None:
                continue
            epoch = int(match.group(1))
            if epoch <= completed_epoch:
                continue

            truncate_at = self._epoch_block_start(lines=lines, header_index=index)
            break

        if truncate_at is None:
            return False

        self.log_path.write_text("".join(lines[:truncate_at]), encoding="utf-8")
        return True

    def purge_tensorboard_after_global_step(self, global_step: int) -> None:
        self._summary_writer.close()
        self._summary_writer = self._build_summary_writer(
            log_dir=str(self.log_path.parent),
            purge_step=global_step + 1,
        )

    @staticmethod
    def _epoch_block_start(lines: list[str], header_index: int) -> int:
        block_start = max(0, header_index - 1)
        if block_start > 0 and lines[block_start - 1].strip() == "":
            block_start -= 1
        return block_start

    def _log_section_header(self, title: str, fill: str) -> None:
        line = fill * self._SECTION_WIDTH
        self.log_message(line)
        self.log_message(f"  {title}")
        self.log_message(line)

    def _log_val_section_header(self, title: str, fill: str) -> None:
        line = fill * self._SECTION_WIDTH
        self.log_val_message(line)
        self.log_val_message(f"  {title}")
        self.log_val_message(line)

    @classmethod
    def _format_lr_group_summary(cls, group: Mapping[str, Any]) -> dict[str, str]:
        nominal_lr = float(group["nominal_lr"])
        effective_lr = float(group["effective_lr"])
        lr = cls._format_lr(nominal_lr)
        if effective_lr != nominal_lr:
            lr = f"{lr}->{cls._format_lr(effective_lr)}"
        return {
            "scope": cls._compact_lr_scope(str(group["lr_scope"])),
            "group": cls._compact_lr_group_name(
                group_name=str(group["group_name"]),
                lr_scope=str(group["lr_scope"]),
            ),
            "lr": lr,
            "params": cls._format_param_count(int(group["num_params"])),
        }

    @staticmethod
    def _compact_lr_scope(lr_scope: str) -> str:
        max_width = 28
        if len(lr_scope) <= max_width:
            return lr_scope
        keep_width = max_width - 3
        left_width = keep_width // 2
        right_width = keep_width - left_width
        return f"{lr_scope[:left_width]}...{lr_scope[-right_width:]}"

    @staticmethod
    def _compact_lr_group_name(group_name: str, lr_scope: str) -> str:
        parts = group_name.split(":")
        if len(parts) >= 3 and parts[0] == lr_scope:
            parts = parts[1:]
        return " ".join(Logger._compact_lr_group_token(part) for part in parts)

    @staticmethod
    def _compact_lr_group_token(token: str) -> str:
        return {
            "no_decay": "no-decay",
        }.get(token, token)

    @staticmethod
    def _format_lr(value: float) -> str:
        return f"{value:.3g}"

    @staticmethod
    def _format_param_count(count: int) -> str:
        abs_count = abs(count)
        for suffix, scale in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
            if abs_count >= scale:
                return f"{count / scale:.3g}{suffix}"
        return str(count)

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
    def write_train_step_scalars(
        self,
        global_step: int | None,
        step_stats: dict[str, float],
        lr: float | None = None,
    ) -> None: ...

    @abstractmethod
    def write_validation_scalars(self, epoch: int, val_metrics: dict[str, float]) -> None: ...

    def write_lr_group_scalars(
        self,
        *,
        epoch: int,
        group_summaries: Sequence[Mapping[str, Any]],
    ) -> None:
        for group in group_summaries:
            self._summary_writer.add_scalar(
                f"Learning_rate/groups/{group['group_name']}",
                float(group["effective_lr"]),
                epoch,
            )
        self._summary_writer.flush()

    def write_trainable_param_scalars(
        self,
        *,
        epoch: int,
        counts: Mapping[str, int],
    ) -> None:
        for name, count in counts.items():
            self._summary_writer.add_scalar(
                f"Trainable_params/{name}",
                int(count),
                epoch,
            )
        self._summary_writer.flush()

    def write_diagnostic_scalars(
        self,
        *,
        epoch: int,
        metrics: Mapping[str, float],
        prefixes: Sequence[str],
    ) -> None:
        selected = self._select_diagnostic_metrics(metrics=metrics, prefixes=prefixes)
        if not selected:
            return

        for key, value in selected.items():
            self._summary_writer.add_scalar(
                self._diagnostic_tensorboard_tag(key),
                float(value),
                epoch,
            )
        self._summary_writer.flush()

    @staticmethod
    def _select_diagnostic_metrics(
        *,
        metrics: Mapping[str, float],
        prefixes: Sequence[str],
    ) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in sorted(metrics.items())
            if any(key.startswith(prefix) for prefix in prefixes)
        }

    @classmethod
    def _diagnostic_tensorboard_tag(cls, key: str) -> str:
        for prefix, tag_prefix in cls._DIAGNOSTIC_TENSORBOARD_PREFIXES.items():
            if key.startswith(prefix):
                return tag_prefix + key[len(prefix) :]
        return "Diagnostics/" + key.replace("/", "_")

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
        purge_step: int | None = None,
    ) -> Any:
        if not self.use_tensorboard:
            summary_writer = _NoOpSummaryWriter()
            return summary_writer
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except Exception:
            summary_writer = _NoOpSummaryWriter()
            return summary_writer
        summary_writer = SummaryWriter(log_dir, purge_step=purge_step)
        return summary_writer
