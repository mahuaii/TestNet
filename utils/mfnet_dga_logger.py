from __future__ import annotations

from typing_extensions import override

from .mfnet_logger import MFNetLogger


class MFNetDGALogger(MFNetLogger):
    @override
    def _write_step_scalars(
        self,
        global_step: int | None,
        step_stats: dict[str, float],
        lr: float | None = None,
    ) -> None:
        super()._write_step_scalars(
            global_step=global_step,
            step_stats=step_stats,
            lr=lr,
        )
        if global_step is None:
            return None

        for key, value in sorted(step_stats.items()):
            if not key.startswith("dga/"):
                continue
            tag = "DGA/" + key[len("dga/") :]
            self._summary_writer.add_scalar(tag, float(value), global_step)
        self._summary_writer.flush()


__all__ = ["MFNetDGALogger"]
