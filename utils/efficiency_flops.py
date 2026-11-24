"""fvcore FLOP analysis with handlers for the operations used by the baselines."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from functools import partial
from importlib.metadata import version
from math import prod
from typing import Any
from unittest.mock import patch

from torch import Tensor, nn


FlopHandle = Callable[[list[Any], list[Any]], int | float | Counter[str]]
FLOP_CONVENTION = "fvcore: fused multiply-add=1; floating-point elementwise estimates; comparisons excluded"
SELECTIVE_SCAN_OP = "prim::PythonOp.SelectiveScanFn"

# Layout changes, eval-mode dropout and comparison-only ops are outside this count.
EXCLUDED_OPS = (
    "aten::adaptive_max_pool2d", "aten::amax", "aten::amin", "aten::clamp_min",
    "aten::dropout_", "aten::expand_as", "aten::feature_dropout", "aten::feature_dropout_",
    "aten::fill_", "aten::flip", "aten::hardtanh", "aten::hardtanh_", "aten::max_pool2d",
    "aten::max_pool2d_with_indices", "aten::max_unpool2d", "aten::ne", "aten::pad",
    "aten::pixel_shuffle", "aten::repeat",
)


def _float_elements(value: Any) -> int:
    from fvcore.nn.jit_handles import get_shape

    if value.type().kind() != "TensorType":
        return 0
    if value.type().scalarType() not in {"Float", "Double", "Half", "BFloat16"}:
        return 0
    shape = get_shape(value)
    if shape is None:
        raise ValueError(f"fvcore FLOP handler requires a known shape: {value}")
    return prod(shape)


def _elementwise(inputs: list[Any], outputs: list[Any], *, cost: int = 1) -> int:
    return cost * _float_elements(outputs[0])


def _gelu(inputs: list[Any], outputs: list[Any]) -> int:
    return _elementwise(inputs, outputs, cost=9 if inputs[1].toIValue() == "tanh" else 5)


def _sum(inputs: list[Any], outputs: list[Any]) -> int:
    return max(0, _float_elements(inputs[0]) - _float_elements(outputs[0]))


def _softmax(inputs: list[Any], outputs: list[Any], *, log: bool) -> int:
    from fvcore.nn.jit_handles import get_shape

    elements = _float_elements(outputs[0])
    groups = elements // get_shape(outputs[0])[inputs[1].toIValue()]
    return 4 * elements if log else 4 * elements - groups


def _vector_norm(inputs: list[Any], outputs: list[Any]) -> int:
    if inputs[1].toIValue() != 2:
        raise ValueError("The fvcore extension supports only L2 vector norms.")
    return 2 * _float_elements(inputs[0])


def _avg_pool2d(inputs: list[Any], outputs: list[Any]) -> int:
    kernel = inputs[1].toIValue()
    if kernel is None:
        kernel = [size.toIValue() for size in inputs[1].node().inputs()]
    if len(kernel) == 1:
        kernel = kernel * 2
    return prod(kernel) * _float_elements(outputs[0])


def _selective_scan(inputs: list[Any], outputs: list[Any]) -> Counter[str]:
    from fvcore.nn.jit_handles import get_shape

    batch, channels, length = get_shape(inputs[0])
    _, state_size = get_shape(inputs[2])
    if len(inputs) != 7 or outputs[0].node().scalar_args() != [None, True, False]:
        raise ValueError("Expected the RS3Mamba selective scan with skip D and no Z gate.")
    # VMamba's prefix-scan estimate, with the D skip term and without the Z gate.
    # https://github.com/MzeroMiko/VMamba/blob/main/classification/models/csms6s.py#L932
    return Counter(selective_scan=batch * channels * length * (9 * state_size + 1))


def _extra_handles() -> dict[str, FlopHandle | None]:
    handles: dict[str, FlopHandle | None] = dict.fromkeys(EXCLUDED_OPS)
    handles.update(
        (f"aten::{name}", _elementwise)
        for name in "add add_ div exp leaky_relu log mul mul_ neg pow reciprocal rsub sqrt square sub".split()
    )
    handles.update({
        "aten::sigmoid": partial(_elementwise, cost=4),
        "aten::silu": partial(_elementwise, cost=5),
        "aten::gelu": _gelu,
        "aten::sum": _sum,
        "aten::mean": lambda inputs, _outputs: _float_elements(inputs[0]),
        "aten::var_mean": lambda inputs, _outputs: 4 * _float_elements(inputs[0]),
        "aten::linalg_vector_norm": _vector_norm,
        "aten::softmax": partial(_softmax, log=False),
        "aten::log_softmax": partial(_softmax, log=True),
        "aten::avg_pool2d": _avg_pool2d,
        "aten::upsample_bicubic2d": partial(_elementwise, cost=16),
        "aten::upsample_linear1d": partial(_elementwise, cost=2),
    })
    return handles


def audit_flops(
    model: nn.Module,
    inputs: tuple[Tensor, ...],
    custom_handles: Mapping[str, FlopHandle | None] | None = None,
    *,
    expects_selective_scan: bool = False,
) -> dict[str, Any]:
    from fvcore.nn import FlopCountAnalysis

    if any("::" not in name for name in custom_handles or ()):
        raise ValueError("fvcore handles require JIT names such as aten::my_op.")

    handles = _extra_handles()
    handles.update(custom_handles or {})
    scan_nodes = 0
    native_calls = 0
    scan_probe = nullcontext()
    if expects_selective_scan:
        import selective_scan_cuda

        native_scan = selective_scan_cuda.fwd
        scan_handle = handles.get(SELECTIVE_SCAN_OP, _selective_scan)
        if scan_handle is None:
            raise ValueError("RS3Mamba selective scan requires a counting handle.")

        def observe_scan(*args: Any, **kwargs: Any) -> Any:
            nonlocal native_calls
            native_calls += 1
            return native_scan(*args, **kwargs)

        def count_scan(args: list[Any], results: list[Any]) -> int | float | Counter[str]:
            nonlocal scan_nodes
            scan_nodes += 1
            return scan_handle(args, results)

        handles[SELECTIVE_SCAN_OP] = count_scan
        scan_probe = patch.object(selective_scan_cuda, "fwd", observe_scan)

    analysis = FlopCountAnalysis(model, inputs).set_op_handle(**handles)
    analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
    with scan_probe:
        total = analysis.total()
    unsupported = dict(sorted(analysis.unsupported_ops().items()))
    if expects_selective_scan and (not native_calls or scan_nodes != native_calls):
        unsupported["selective_scan_cuda::unmatched_fwd"] = max(1, abs(native_calls - scan_nodes))

    return {
        "backend": "fvcore",
        "backend_version": version("fvcore"),
        "convention": FLOP_CONVENTION,
        "complete": not unsupported,
        "counted_flops": total,
        "unsupported_ops": unsupported,
        "by_operator": dict(sorted(analysis.by_operator().items())),
        "uncalled_modules": sorted(analysis.uncalled_modules()),
        "native_selective_scan_calls": native_calls,
        "traced_selective_scan_nodes": scan_nodes,
    }
