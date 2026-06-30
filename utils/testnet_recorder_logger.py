from __future__ import annotations

from typing_extensions import override

from .testnet_logger import TestNetLogger


_TENSORBOARD_PREFIXES = {
    "dga/": "DGA/",
    "dgsf10/": "DGSF10/",
    "prealign/": "PreAlign/",
    "spmf20/structure/": "SPMF20/Structure/",
    "spmf20/": "SPMF20/",
}


class TestNetRecorderLogger(TestNetLogger):
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
            tag = self._tensorboard_tag(key)
            if tag is not None:
                self._summary_writer.add_scalar(tag, float(value), global_step)
        self._summary_writer.flush()

    @staticmethod
    def _tensorboard_tag(key: str) -> str | None:
        for metric_prefix, tag_prefix in _TENSORBOARD_PREFIXES.items():
            if key.startswith(metric_prefix):
                return tag_prefix + key[len(metric_prefix) :]
        return None


__all__ = ["TestNetRecorderLogger"]
