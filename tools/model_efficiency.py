#!/usr/bin/env python
"""Measure full dual-modal inference with common FP32 settings; see tools/model_efficiency.md."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import importlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config, resolve_config_path
from utils.efficiency_adapters import EfficiencyAdapter, build_repository_adapter
from utils.efficiency_flops import FLOP_CONVENTION, audit_flops

CSV_FIELDS = [
    "run_id", "timestamp_utc", "model_name", "model_type", "config", "checkpoint",
    "adapter", "task_weights", "train_epoch", "stage_name", "freeze_policy",
    "total_params_m", "trainable_params_m", "gflops", "counted_gflops",
    "flops_complete", "unsupported_ops", "latency_ms_per_patch",
    "latency_std_ms_per_patch", "height", "width", "batch_size", "precision",
    "device", "device_name", "reference_rtx3090", "warmup", "iterations", "seed",
    "tf32", "cudnn_benchmark", "cudnn_deterministic", "deterministic_algorithms",
    "cpu_threads", "timing_method", "flop_convention", "torch_version", "cuda_version",
    "cudnn_version", "report_path", "settings_json",
]


def import_symbol(spec: str) -> Any:
    module, name = spec.split(":", maxsplit=1)
    return getattr(importlib.import_module(module), name)


def parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    # parameters() removes repeated references to the same Parameter, including
    # MFNet's sam.image_encoder / image_encoder alias. Keep frozen parameters.
    parameters = tuple(model.parameters())
    return sum(p.numel() for p in parameters), sum(p.numel() for p in parameters if p.requires_grad)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency(
    adapter: EfficiencyAdapter, rgb: torch.Tensor, dsm: torch.Tensor,
    device: torch.device, warmup: int, iterations: int,
) -> list[float]:
    for _ in range(warmup):
        output = adapter(rgb, dsm)
        del output
    synchronize(device)
    samples = []
    for _ in range(iterations):
        synchronize(device)
        start = time.perf_counter_ns()
        output = adapter(rgb, dsm)
        synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1e6 / rgb.shape[0])
        del output
    return samples


def configure_runtime(device: torch.device, seed: int) -> dict[str, Any]:
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Supported devices are a single CUDA GPU or CPU.")
    if device.type == "cuda":
        device = torch.device("cuda", 0 if device.index is None else device.index)
        torch.cuda.set_device(device)
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = platform.processor() or platform.machine()
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    return {
        "device": str(device), "device_name": device_name,
        "reference_rtx3090": device.type == "cuda" and "RTX 3090" in device_name and "Ti" not in device_name,
        "precision": "float32", "tf32": False, "cudnn_benchmark": False,
        "cudnn_deterministic": False, "deterministic_algorithms": False,
        "cpu_threads": torch.get_num_threads(), "seed": seed,
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "timing_method": "perf_counter_ns; synchronize selected CUDA GPU before/after each forward",
        "flop_convention": FLOP_CONVENTION,
        "input_layout": "RGB BCHW (3 channels), DSM BHW (1 channel)",
        "input_values": "seeded uniform [0,1), generated on CPU then copied before timing",
        "timing_scope": "adapter+full model; includes required conversion/resize; excludes IO/H2D/output disposal",
        "execution": "eager, eval, inference_mode, autocast disabled, contiguous FP32",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_model(job: dict[str, Any], args: argparse.Namespace, runtime: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config(job["config"])
    if job["model_type"] is not None:
        cfg["model"]["type"] = job["model_type"]
    options = cfg.get("efficiency", {})
    adapter_spec = job["adapter"] or options.get("adapter", "repository")
    factory = build_repository_adapter if adapter_spec == "repository" else import_symbol(adapter_spec)
    epoch = args.train_epoch if args.train_epoch is not None else options.get("train_epoch")
    device = torch.device(runtime["device"])
    # Reset model initialization and synthetic inputs for every model.
    torch.manual_seed(args.seed)
    adapter = factory(cfg, job["checkpoint"], epoch)
    if not isinstance(adapter, EfficiencyAdapter):
        raise TypeError("Adapter factory must return an EfficiencyAdapter.")
    adapter.to(device=device, dtype=torch.float32).eval()
    total, trainable = parameter_counts(adapter.model)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    rgb = torch.rand(args.batch_size, 3, *args.input_size, generator=generator, dtype=torch.float32).to(device)
    dsm = torch.rand(args.batch_size, *args.input_size, generator=generator, dtype=torch.float32).to(device)
    settings = {
        **runtime, "height": args.input_size[0], "width": args.input_size[1],
        "batch_size": args.batch_size, "warmup": args.warmup, "iterations": args.iterations,
        "custom_flop_rules": args.flop_rules,
    }
    print(f"[{job['name']}] {adapter.metadata['model_type']} on {runtime['device_name']}", flush=True)
    custom_rules = {} if args.flop_rules is None else import_symbol(args.flop_rules)()
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        samples = measure_latency(adapter, rgb, dsm, device, args.warmup, args.iterations)
        flops = audit_flops(
            adapter, (rgb, dsm), custom_rules,
            expects_selective_scan=adapter.metadata["model_type"] == "rs3mamba",
        )
        synchronize(device)
    counted_gflops = flops["counted_flops"] / args.batch_size / 1e9
    report_path = args.output.parent / f"{args.output.stem}_{job['run_id']}.json"
    row = {
        **{key: value for key, value in settings.items() if key in CSV_FIELDS},
        "run_id": job["run_id"], "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": job["name"], "model_type": adapter.metadata["model_type"],
        "config": str(job["config"]),
        "checkpoint": str(job["checkpoint"]) if job["checkpoint"] is not None else "",
        "adapter": adapter_spec, "task_weights": adapter.metadata["task_weights"],
        "train_epoch": adapter.metadata.get("train_epoch"),
        "stage_name": adapter.metadata.get("stage_name"),
        "freeze_policy": adapter.metadata["freeze_policy"],
        "total_params_m": total / 1e6, "trainable_params_m": trainable / 1e6,
        "gflops": counted_gflops if flops["complete"] else "",
        "counted_gflops": counted_gflops, "flops_complete": flops["complete"],
        "unsupported_ops": json.dumps(flops["unsupported_ops"], sort_keys=True),
        "latency_ms_per_patch": statistics.mean(samples),
        "latency_std_ms_per_patch": statistics.pstdev(samples),
        "report_path": str(report_path), "settings_json": json.dumps(settings, sort_keys=True),
    }
    report = {
        "result": row, "settings": settings, "effective_config": cfg,
        "adapter_metadata": adapter.metadata, "total_parameters": total,
        "trainable_parameters": trainable, "flop_audit": flops,
        "latency_samples_ms_per_patch": samples,
        "parameter_policy": "all model Parameters including frozen and unused modules; shared Parameter identity deduplicated",
        "flops_scope": "one fvcore-traced inference path including adapter; normalized per patch",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"  total={row['total_params_m']:.3f}M trainable={row['trainable_params_m']:.3f}M "
        f"GFLOPs={row['gflops'] if flops['complete'] else 'INCOMPLETE'} "
        f"latency={row['latency_ms_per_patch']:.3f} ms/patch", flush=True,
    )
    if not flops["complete"]:
        print(f"  Unsupported operators: {row['unsupported_ops']}; details: {report_path}", flush=True)
    del adapter, rgb, dsm
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True, help="Existing training/effective JSON(C) configs")
    parser.add_argument("--checkpoint", nargs="+", type=Path, help="One full task checkpoint per config; omitted = initialization weights")
    parser.add_argument("--model-name", nargs="+", help="CSV display names; architecture still follows config/model-type")
    parser.add_argument("--model-type", nargs="+", help="Override config.model.type, one per config")
    parser.add_argument("--adapter", nargs="+", help="repository or module:factory, one per config")
    parser.add_argument("--input-size", nargs=2, type=int, default=[256, 256], metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--train-epoch", type=int, help="Training freeze stage; default checkpoint epoch or efficiency.train_epoch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flop-rules", help="module:factory returning fvcore JIT operator handles")
    parser.add_argument("--output", type=Path, default=Path("work_dirs/model_efficiency.csv"), help="CSV destination; existing file is replaced")
    args = parser.parse_args()
    for name in ("checkpoint", "model_name", "model_type", "adapter"):
        values = getattr(args, name)
        if values is not None and len(values) != len(args.config):
            parser.error(f"--{name.replace('_', '-')} must contain one value per --config")
    if min(*args.input_size, args.batch_size, args.iterations) < 1 or args.warmup < 0:
        parser.error("Input dimensions, batch size and iterations must be positive; warmup must be nonnegative")
    if args.train_epoch is not None and args.train_epoch < 1:
        parser.error("--train-epoch must be positive and 1-based")
    args.output = args.output.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    runtime = configure_runtime(torch.device(args.device), args.seed)
    if not runtime["reference_rtx3090"]:
        print(f"Device is {runtime['device_name']}; results will record this hardware instead of RTX 3090.", flush=True)
    jobs = []
    for index, config in enumerate(args.config):
        path = resolve_config_path(config)
        jobs.append({
            "run_id": uuid.uuid4().hex,
            "config": path,
            "name": args.model_name[index] if args.model_name else path.stem,
            "checkpoint": args.checkpoint[index].expanduser().resolve(strict=True) if args.checkpoint else None,
            "model_type": args.model_type[index] if args.model_type else None,
            "adapter": args.adapter[index] if args.adapter else None,
        })
    rows = []
    for job in jobs:
        rows.append(run_model(job, args, runtime))
        # Persist completed models even if a subsequent model fails.
        write_csv(args.output, rows)
    print(f"Saved {args.output}", flush=True)
    if any(not row["flops_complete"] for row in rows):
        raise SystemExit("Incomplete FLOP coverage: results saved; add rules using --flop-rules before reporting full GFLOPs.")


if __name__ == "__main__":
    main()
