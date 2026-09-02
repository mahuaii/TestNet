from __future__ import annotations

import torch


def configure_3090_runtime(
    device: torch.device,
    *,
    enable_tf32: bool = True,
) -> None:
    """Configure and validate the CUDA runtime used by the 3090 entrypoint."""
    if device.type != "cuda":
        raise ValueError(f"The 3090 training entrypoint requires a CUDA device, got {device}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the 3090 training entrypoint cannot continue.")

    if device.index is not None:
        torch.cuda.set_device(device)

    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8:
        raise RuntimeError(
            "The 3090 training entrypoint requires an Ampere-or-newer CUDA device "
            f"with BF16 support, got compute capability {capability}."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support BF16 autocast.")

    if not isinstance(enable_tf32, bool):
        raise TypeError("runtime_3090.tf32 must be a boolean")

    torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32


__all__ = ["configure_3090_runtime"]
