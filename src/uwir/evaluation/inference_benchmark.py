"""Consistent network-only and end-to-end inference profiling."""

from __future__ import annotations

import time
from typing import Any

import torch


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark_inference(adapter, *, warmup=10, iterations=50, size=256) -> dict[str, Any]:
    adapter.eval()
    device = adapter.device
    sample = torch.rand(1, 3, size, size, device=device)
    inference_module, network_inputs = adapter.network_benchmark_call(sample)
    for _ in range(warmup):
        adapter.inference(sample)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(iterations):
        adapter.inference(sample)
    _synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000 / iterations
    for _ in range(warmup):
        with adapter.autocast():
            inference_module(*network_inputs)
    _synchronize(device)
    network_start = time.perf_counter()
    for _ in range(iterations):
        with adapter.autocast():
            inference_module(*network_inputs)
    _synchronize(device)
    network_elapsed = (time.perf_counter() - network_start) * 1000 / iterations
    parameters = sum(parameter.numel() for parameter in inference_module.parameters())
    flops = None
    try:
        from thop import profile

        flops = float(profile(inference_module, inputs=network_inputs, verbose=False)[0])
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {
        "method": adapter.display_name,
        "input_shape": [1, 3, size, size],
        "inference_parameters": parameters,
        "flops": flops,
        "network_only_latency_ms": network_elapsed,
        "end_to_end_latency_ms": elapsed,
        "peak_gpu_memory_bytes": peak,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "discriminator_excluded": adapter.method_name == "funie_gan",
    }
