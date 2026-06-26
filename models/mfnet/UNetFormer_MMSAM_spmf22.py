from __future__ import annotations

from .UNetFormer_MMSAM_spmf import UNetFormerSPMF, _as_single_channel_dsm


class UNetFormerSPMF22(UNetFormerSPMF):
    spmf_variant = "22"


__all__ = ["UNetFormerSPMF22", "_as_single_channel_dsm"]
