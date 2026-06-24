from __future__ import annotations

from .UNetFormer_MMSAM_spmf import UNetFormerSPMF, _as_single_channel_dsm


class UNetFormerSPMF11(UNetFormerSPMF):
    spmf_variant = "11"


__all__ = ["UNetFormerSPMF11", "_as_single_channel_dsm"]
