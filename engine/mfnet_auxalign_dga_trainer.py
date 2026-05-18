from __future__ import annotations

from utils.stat_tracker import StatTracker

from .mfnet_auxalign_trainer import MFNetAuxAlignTrainer
from .mfnet_dga_trainer import MFNetDGATrainer


class MFNetAuxAlignDGATrainer(MFNetAuxAlignTrainer):
    def after_step(
        self,
        step: int,
        step_stats_tracker: StatTracker,
        is_last_step_of_epoch: bool = False,
    ) -> None:
        return MFNetDGATrainer.after_step(self, step, step_stats_tracker, is_last_step_of_epoch)

    _collect_dga_block_scalars = MFNetDGATrainer._collect_dga_block_scalars
    _gate_scalar = staticmethod(MFNetDGATrainer._gate_scalar)
    _to_float = staticmethod(MFNetDGATrainer._to_float)


__all__ = ["MFNetAuxAlignDGATrainer"]
