from __future__ import annotations

import torch.nn as nn

from .UNetFormer_MMSAM_prealign_spmf21 import UNetFormerPreAlignSPMF21


class UNetFormerPreAlignSPMF21DFM(UNetFormerPreAlignSPMF21):
    """GRC + GSMM ablation with additive structure injection and original DFM."""

    spmf_variant = "21_dfm"

    def _build_spmf_fusion(self) -> nn.Module:
        return self._spmf_variant_spec.fusion_cls(channels=256)


__all__ = ["UNetFormerPreAlignSPMF21DFM"]
