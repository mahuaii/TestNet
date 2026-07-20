from __future__ import annotations

from .UNetFormer_MMSAM_spmf import UNetFormerSPMF, _as_single_channel_dsm


class UNetFormerSPMF30(UNetFormerSPMF):
    spmf_variant = "30"


__all__ = ["UNetFormerSPMF30", "_as_single_channel_dsm"]
