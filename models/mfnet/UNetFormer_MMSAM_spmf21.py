from __future__ import annotations

from .UNetFormer_MMSAM_spmf import UNetFormerSPMF, _as_single_channel_dsm


class UNetFormerSPMF21(UNetFormerSPMF):
    spmf_variant = "21"


__all__ = ["UNetFormerSPMF21", "_as_single_channel_dsm"]
