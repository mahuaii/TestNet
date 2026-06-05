import torch
import torch.nn as nn


def _validate_bhwc_pair(rgb, dsm, dims):
    if rgb.shape != dsm.shape:
        raise ValueError(
            f"Expected rgb and dsm to have the same shape, got {tuple(rgb.shape)} and {tuple(dsm.shape)}."
        )
    if rgb.ndim != 4:
        raise ValueError(f"Expected rgb and dsm to be 4D BHWC tensors, got shape {tuple(rgb.shape)}.")
    if rgb.shape[-1] != dims:
        raise ValueError(f"Expected input channel count {dims}, got {rgb.shape[-1]}.")


class DGFM01(nn.Module):
    """MoBaNet DGFM core with TestNet BHWC-in/BCHW-out boundaries."""

    def __init__(self, dims, ratio=0.25):
        super().__init__()
        self.dims = int(dims)
        hidden_dim = max(32, int(dims * min(ratio, 0.125)))

        self.rgb_reduce = nn.Conv2d(dims, hidden_dim, kernel_size=1, bias=False)
        self.dsm_reduce = nn.Conv2d(dims, hidden_dim, kernel_size=1, bias=False)
        self.gate_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dims, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        nn.init.zeros_(self.gate_net[-2].weight)
        nn.init.zeros_(self.gate_net[-2].bias)

    def forward(self, rgb, dsm):
        _validate_bhwc_pair(rgb, dsm, self.dims)
        rgb = rgb.permute(0, 3, 1, 2).contiguous()
        dsm = dsm.permute(0, 3, 1, 2).contiguous()

        rgb_reduce = self.rgb_reduce(rgb)
        dsm_reduce = self.dsm_reduce(dsm)
        diff_reduce = torch.abs(rgb_reduce - dsm_reduce)

        gate = self.gate_net(torch.cat([rgb_reduce, dsm_reduce, diff_reduce], dim=1))
        return gate * rgb + (1.0 - gate) * dsm


__all__ = ["DGFM01"]
