from __future__ import annotations

from .UNetFormer_MMSAM_spmf import UNetFormerSPMF, _as_single_channel_dsm


class UNetFormerSPMF20(UNetFormerSPMF):
    spmf_variant = "20"


__all__ = ["UNetFormerSPMF20", "_as_single_channel_dsm"]
