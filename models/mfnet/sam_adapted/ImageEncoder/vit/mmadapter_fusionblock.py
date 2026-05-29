import torch
import torch.nn as nn

from .adapter_fusionblock import AdapterFusionBlock


class _DepthwiseLocalAdapter(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 0.25) -> None:
        super().__init__()
        hidden_dim = max(1, int(dim * mlp_ratio))
        self.D_fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.local_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.D_fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.local_conv(xs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.D_fc2(xs)


class MMAdapter10FusionBlock(AdapterFusionBlock):
    def __init__(self, *args, **kwargs) -> None:
        dim = kwargs.get("dim")
        if dim is None and len(args) >= 2:
            dim = args[1]
        super().__init__(*args, **kwargs)
        del self.wx_Adapter
        del self.wy_Adapter
        self.MMAdapter_Fusion = nn.Sequential(
            nn.Linear(2 * int(dim), 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def fuse_adapter_messages(
        self,
        x_msg: torch.Tensor,
        y_msg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x_msg.ndim != 4 or y_msg.ndim != 4:
            raise ValueError(
                "MMAdapter10FusionBlock expects x_msg and y_msg with shape [B, H, W, C]."
            )
        if x_msg.shape != y_msg.shape:
            raise ValueError(
                "MMAdapter10FusionBlock expects x_msg and y_msg to have the same shape."
            )

        z = torch.cat([x_msg, y_msg], dim=-1)
        gate = self.MMAdapter_Fusion(z)
        gate_y_to_x = gate[..., 0:1]
        gate_x_to_y = gate[..., 1:2]
        x_fuse = x_msg + gate_y_to_x * y_msg
        y_fuse = y_msg + gate_x_to_y * x_msg
        return x_fuse, y_fuse


class MMAdapter20FusionBlock(AdapterFusionBlock):
    def __init__(self, *args, alpha_init: float = 1e-3, **kwargs) -> None:
        dim = kwargs.get("dim")
        if dim is None and len(args) >= 2:
            dim = args[1]
        super().__init__(*args, **kwargs)
        del self.wx_Adapter
        del self.wy_Adapter
        self.MMAdapter_Fusion = nn.Linear(int(dim), 1)
        self.MMAdapter_alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    @property
    def alpha(self) -> nn.Parameter:
        return self.MMAdapter_alpha

    @property
    def gate_proj(self) -> nn.Linear:
        return self.MMAdapter_Fusion

    def fuse_adapter_messages(
        self,
        x_msg: torch.Tensor,
        y_msg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x_msg.ndim != 4 or y_msg.ndim != 4:
            raise ValueError(
                "MMAdapter20FusionBlock expects x_msg and y_msg with shape [B, H, W, C]."
            )
        if x_msg.shape != y_msg.shape:
            raise ValueError(
                "MMAdapter20FusionBlock expects x_msg and y_msg to have the same shape."
            )

        gate = torch.sigmoid(self.gate_proj(y_msg))
        x_fuse = self.MMAdapter_alpha * gate * x_msg
        y_fuse = torch.zeros_like(y_msg)
        return x_fuse, y_fuse


class MMAdapter21FusionBlock(MMAdapter20FusionBlock):
    def __init__(self, *args, **kwargs) -> None:
        dim = kwargs.get("dim")
        if dim is None and len(args) >= 2:
            dim = args[1]
        super().__init__(*args, **kwargs)
        self.MLPy_Adapter = _DepthwiseLocalAdapter(int(dim))

    def fuse_adapter_messages(
        self,
        x_msg: torch.Tensor,
        y_msg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x_msg.ndim != 4 or y_msg.ndim != 4:
            raise ValueError(
                "MMAdapter21FusionBlock expects x_msg and y_msg with shape [B, H, W, C]."
            )
        if x_msg.shape != y_msg.shape:
            raise ValueError(
                "MMAdapter21FusionBlock expects x_msg and y_msg to have the same shape."
            )

        gate = torch.sigmoid(self.gate_proj(y_msg))
        x_fuse = self.MMAdapter_alpha * gate * x_msg
        y_fuse = torch.zeros_like(y_msg)
        return x_fuse, y_fuse


__all__ = ["MMAdapter10FusionBlock", "MMAdapter20FusionBlock", "MMAdapter21FusionBlock"]
