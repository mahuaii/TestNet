from __future__ import annotations

from .UNetFormer_MMSAM_spmf import UNetFormerSPMF, _as_single_channel_dsm


class UNetFormerSPMF10(UNetFormerSPMF):
    spmf_variant = "10"


__all__ = ["UNetFormerSPMF10", "_as_single_channel_dsm"]
