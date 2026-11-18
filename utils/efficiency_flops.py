"""Audit the eager forward's dispatched operators, without replacing the model.

Rules estimate algorithmic FP32 operations, not GPU instructions. A MAC is two
FLOPs. Comparisons used by activations/pooling count as one operation. See
tools/model_efficiency.md for the complete convention and extension contract.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from math import ceil, floor, prod
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

FlopRule = Callable[[dict[str, Any], Any], int]
FLOP_CONVENTION = "algorithmic_v1: MAC=2; scalar/special/comparison=1; interpolation estimated"


def _output(out: Any) -> torch.Tensor:
    return out if isinstance(out, torch.Tensor) else out[0]


def _elements(a: dict[str, Any], out: Any, cost: int = 1) -> int:
    return _output(out).numel() * cost


def _add(a: dict[str, Any], out: Any) -> int:
    return out.numel() * (1 + int(a.get("alpha", 1) != 1))


def _conv(a: dict[str, Any], out: Any) -> int:
    x, weight = a["input"], a["weight"]
    spatial = x.shape[2:] if a["transposed"] else out.shape[2:]
    # Weight already contains in_channels/groups (or out_channels/groups).
    return 2 * x.shape[0] * prod(spatial) * weight.numel() + (out.numel() if a["bias"] is not None else 0)


def _mm(a: dict[str, Any], out: Any) -> int:
    return 2 * out.numel() * a["self"].shape[-1]


def _addmm(a: dict[str, Any], out: Any) -> int:
    return (
        2 * out.numel() * a["mat1"].shape[-1]
        + out.numel() * (int(a["beta"] != 0) + int(a["beta"] not in (0, 1)) + int(a["alpha"] != 1))
    )


def _baddbmm(a: dict[str, Any], out: Any) -> int:
    return (
        2 * out.numel() * a["batch1"].shape[-1]
        + out.numel() * (int(a["beta"] != 0) + int(a["beta"] not in (0, 1)) + int(a["alpha"] != 1))
    )


def _linear(a: dict[str, Any], out: Any) -> int:
    return 2 * out.numel() * a["weight"].shape[-1] + (out.numel() if a["bias"] is not None else 0)


def _reduction(a: dict[str, Any], out: Any, mean: bool = False) -> int:
    return max(0, a["self"].numel() - _output(out).numel()) + (int(mean) * _output(out).numel())


def _softmax(a: dict[str, Any], out: Any) -> int:
    x = a["self"]
    groups = x.numel() // x.shape[a["dim"]]
    # Max reduction, subtraction, exp, sum reduction, division.
    return 5 * x.numel() - 2 * groups


def _log_softmax(a: dict[str, Any], out: Any) -> int:
    x = a["self"]
    return 5 * x.numel() - x.numel() // x.shape[a["dim"]]


def _layer_norm(a: dict[str, Any], out: Any) -> int:
    x = a["input"]
    groups = x.numel() // prod(a["normalized_shape"])
    return 5 * x.numel() + 2 * groups + x.numel() * (int(a["weight"] is not None) + int(a["bias"] is not None))


def _batch_norm(a: dict[str, Any], out: Any) -> int:
    x = a["input"]
    if a.get("training", False):
        raise ValueError("FLOP rule expects BatchNorm with running statistics in eval mode.")
    return 2 * x.numel() + 2 * x.shape[1] + x.numel() * (int(a["weight"] is not None) + int(a["bias"] is not None))


def _pool(a: dict[str, Any], out: Any, average: bool) -> int:
    kernel = a["kernel_size"]
    volume = prod(kernel) if isinstance(kernel, (tuple, list)) else kernel
    # Dense-window convention, including boundary padding.
    return _output(out).numel() * (volume if average else max(0, volume - 1))


def _adaptive_pool(a: dict[str, Any], out: Any) -> int:
    x, y = a["self"], _output(out)
    # Adaptive bins may overlap for non-divisible spatial shapes.
    visits = prod(
        sum(ceil((i + 1) * src / dst) - floor(i * src / dst) for i in range(dst))
        for src, dst in zip(x.shape[2:], y.shape[2:])
    )
    return x.shape[0] * x.shape[1] * visits


def _attention(a: dict[str, Any], out: Any) -> int:
    q, k, v = a["query"], a["key"], a["value"]
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Fused attention rule requires dense BHLD tensors.")
    dropout = a.get("dropout_p", 0.0)
    if dropout != 0:
        raise ValueError("Inference attention must use dropout_p=0.")
    groups = q.shape[0] * q.shape[1] * q.shape[2]
    scores = groups * k.shape[2]
    # Dense QK + AV, score scaling, stable softmax; causal uses dense convention.
    mask = a.get("attn_mask", a.get("attn_bias"))
    return 2 * scores * (q.shape[-1] + v.shape[-1]) + 6 * scores - 2 * groups + (scores if mask is not None else 0)


def default_rules() -> dict[str, FlopRule]:
    rules: dict[str, FlopRule] = {}

    def register(names: str, rule: FlopRule) -> None:
        for name in names.split():
            rules[f"aten.{name}"] = rule

    register("convolution _convolution", _conv)
    register("mm bmm matmul mv dot", _mm)
    register("addmm", _addmm)
    register("baddbmm", _baddbmm)
    register("linear", _linear)
    register("add add_ sub sub_ rsub", _add)
    register("mul mul_ div div_ true_divide pow square neg abs exp log log2 sqrt rsqrt reciprocal tanh relu relu_ clamp_min clamp_max minimum maximum", _elements)
    register("clamp", lambda a, out: _elements(a, out, int(a["min"] is not None) + int(a["max"] is not None)))
    register("sigmoid sigmoid_", lambda a, out: _elements(a, out, 4))
    register("silu silu_", lambda a, out: _elements(a, out, 5))
    register("gelu gelu_", lambda a, out: _elements(a, out, 9 if a["approximate"] == "tanh" else 5))
    register("leaky_relu leaky_relu_", lambda a, out: _elements(a, out, 2))
    register("hardtanh hardtanh_", lambda a, out: _elements(a, out, 2))
    register("hardswish hardswish_", lambda a, out: _elements(a, out, 4))
    register("hardsigmoid hardsigmoid_", lambda a, out: _elements(a, out, 3))
    register("sum amax amin", _reduction)
    register("mean", lambda a, out: _reduction(a, out, mean=True))
    register("_softmax softmax", _softmax)
    register("_log_softmax log_softmax", _log_softmax)
    register("native_layer_norm layer_norm", _layer_norm)
    register("native_batch_norm _native_batch_norm_legit _native_batch_norm_legit_no_training batch_norm cudnn_batch_norm miopen_batch_norm", _batch_norm)
    register("avg_pool1d avg_pool2d avg_pool3d", lambda a, out: _pool(a, out, True))
    register("max_pool1d max_pool2d max_pool3d max_pool2d_with_indices max_pool3d_with_indices", lambda a, out: _pool(a, out, False))
    register("_adaptive_avg_pool2d adaptive_avg_pool1d adaptive_avg_pool2d adaptive_avg_pool3d _adaptive_avg_pool3d", _adaptive_pool)
    register("upsample_linear1d", lambda a, out: _elements(a, out, 3))
    register("upsample_bilinear2d", lambda a, out: _elements(a, out, 7))
    register("upsample_bicubic2d", lambda a, out: _elements(a, out, 31))
    register("upsample_trilinear3d", lambda a, out: _elements(a, out, 15))
    register("scaled_dot_product_attention _scaled_dot_product_flash_attention _scaled_dot_product_flash_attention_for_cpu _scaled_dot_product_efficient_attention _scaled_dot_product_cudnn_attention", _attention)
    return rules


# Explicit zero-arithmetic policy. Memory work still belongs in latency timing.
ZERO_ARITHMETIC = {
    f"aten.{name}" for name in """
    alias detach detach_ view view_as _unsafe_view reshape reshape_as flatten
    permute transpose t squeeze squeeze_ unsqueeze unsqueeze_ expand expand_as
    contiguous clone copy_ _to_copy to type_as lift_fresh lift_fresh_copy
    slice select narrow split split_with_sizes unbind chunk as_strided
    cat stack repeat repeat_interleave tile index index_select gather take
    constant_pad_nd reflection_pad1d reflection_pad2d reflection_pad3d
    replication_pad1d replication_pad2d replication_pad3d
    empty empty_like empty_strided new_empty new_empty_strided zeros zeros_like
    ones ones_like full full_like new_zeros new_ones new_full fill_ zero_
    arange linspace scalar_tensor _local_scalar_dense item
    eq ne lt le gt ge logical_and logical_or logical_not bitwise_and bitwise_or
    is_nonzero isfinite isinf isnan where masked_fill masked_fill_
    upsample_nearest1d upsample_nearest2d upsample_nearest3d
    _upsample_nearest_exact1d _upsample_nearest_exact2d _upsample_nearest_exact3d
    """.split()
}


class ForwardFlopAudit(TorchDispatchMode):
    """Every dispatched operator is counted, explicitly excluded or unsupported.

    Extensions map exact overload names (preferred) or packet names to
    ``rule(bound_schema_arguments, output) -> int``. Rules use tensor shapes,
    never call the model or run tensor arithmetic themselves. Returning zero
    explicitly excludes an operator; document why in the custom rule's source.
    """

    def __init__(self, custom_rules: dict[str, FlopRule] | None = None) -> None:
        super().__init__()
        self.rules = default_rules()
        self.rules.update(custom_rules or {})
        self.calls: Counter[str] = Counter()
        self.flops: Counter[str] = Counter()
        self.status: dict[str, str] = {}
        self.reasons: dict[str, str] = {}
        self.input_shapes: dict[str, list[Any]] = defaultdict(list)

    def __torch_dispatch__(self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name, packet = str(func), str(func._overloadpacket)
        self.calls[name] += 1
        bound = {
            arg.name: args[i] if i < len(args) else kwargs.get(arg.name, arg.default_value)
            for i, arg in enumerate(func._schema.arguments)
        }
        rule = self.rules.get(name, self.rules.get(packet))
        if rule is not None:
            count = int(rule(bound, out))
            if count < 0:
                raise ValueError(f"Negative FLOPs from rule for {name}.")
            self.flops[name] += count
            self.status[name] = "counted"
        elif packet in ZERO_ARITHMETIC:
            self.status[name] = "excluded"
            self.reasons[name] = "zero floating arithmetic by policy; memory/index/comparison work remains in latency"
        else:
            self.status[name] = "unsupported"
            shapes = {key: list(value.shape) for key, value in bound.items() if isinstance(value, torch.Tensor)}
            if shapes not in self.input_shapes[name] and len(self.input_shapes[name]) < 5:
                self.input_shapes[name].append(shapes)
        return out

    def report(self) -> dict[str, Any]:
        unsupported = {name: self.calls[name] for name in sorted(self.calls) if self.status[name] == "unsupported"}
        return {
            "convention": FLOP_CONVENTION,
            "complete": not unsupported,
            "counted_flops": sum(self.flops.values()),
            "unsupported_ops": unsupported,
            "operators": {
                name: {
                    "calls": self.calls[name], "flops": self.flops[name],
                    "status": self.status[name], "reason": self.reasons.get(name),
                    "unsupported_input_shapes": self.input_shapes.get(name, []),
                } for name in sorted(self.calls)
            },
        }
