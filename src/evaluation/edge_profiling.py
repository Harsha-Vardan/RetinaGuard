"""RetinaGuard - Edge-Device Profiling (Latency / RAM / Params / FLOPs).

Benchmarks model readiness for low-cost tele-screening on CPU / edge devices
(e.g. Remidio Fundus on Phone). Reports parameter count, FLOPs, inference
latency and peak RAM consumption.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import time
import tracemalloc

import numpy as np
import torch
import torch.nn as nn

try:
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    _HAS_PSUTIL = False


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "params": int(total),
        "n_params": int(total),
        "parameter_count": int(total),
    }


def _params_millions(model: nn.Module) -> float:
    return count_parameters(model)["total_params"] / 1e6


def estimate_flops(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
) -> float:
    """Estimate FLOPs (returns number of FLOPs, not GFLOPs). Best-effort."""
    # Try thop
    try:
        from thop import profile as thop_profile  # type: ignore

        dummy = torch.randn(*input_size, device=device)
        model = model.to(device).eval()
        macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        return float(macs * 2)  # MACs -> FLOPs
    except Exception:
        pass

    # Try fvcore
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        dummy = torch.randn(*input_size, device=device)
        model = model.to(device).eval()
        fca = FlopCountAnalysis(model, dummy)
        return float(fca.total())
    except Exception:
        pass

    # Try torch profiler with_flops
    try:
        dummy = torch.randn(*input_size, device=device)
        model = model.to(device).eval()
        with torch.profiler.profile(with_flops=True) as prof:
            with torch.no_grad():
                model(dummy)
        total_flops = 0
        for evt in prof.key_averages():
            if hasattr(evt, "flops") and evt.flops is not None:
                total_flops += evt.flops
        if total_flops > 0:
            return float(total_flops)
    except Exception:
        pass

    # Heuristic fallback: ~2 FLOPs per parameter per pixel-ish; use params * 2 * (H*W / (224*224)?)
    # Simpler: params * 2 (rough). Return params*2.
    return float(count_parameters(model)["total_params"] * 2)


def measure_latency(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
    n_warmup: int = 3,
    n_runs: int = 10,
    batch_size: Optional[int] = None,
) -> Dict[str, float]:
    """Measure inference latency on the given device."""
    if batch_size is not None:
        input_size = (batch_size,) + tuple(input_size[1:])
    model = model.to(device).eval()
    dummy = torch.randn(*input_size, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            try:
                model(dummy)
            except Exception:
                pass

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            try:
                model(dummy)
            except Exception:
                pass
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    times_arr = np.array(times, dtype=float)
    mean_ms = float(np.mean(times_arr)) if len(times_arr) else 0.0
    return {
        "latency_ms": round(mean_ms, 4),
        "latency_mean_ms": round(mean_ms, 4),
        "latency_std_ms": round(float(np.std(times_arr)), 4) if len(times_arr) else 0.0,
        "latency_min_ms": round(float(np.min(times_arr)), 4) if len(times_arr) else 0.0,
        "latency_max_ms": round(float(np.max(times_arr)), 4) if len(times_arr) else 0.0,
        "latency_p95_ms": round(float(np.percentile(times_arr, 95)), 4) if len(times_arr) else 0.0,
        "fps": round(1000.0 / mean_ms, 4) if mean_ms > 0 else 0.0,
        "throughput_fps": round(1000.0 / mean_ms, 4) if mean_ms > 0 else 0.0,
        "n_runs": int(n_runs),
    }


def measure_peak_ram(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
) -> Dict[str, float]:
    """Measure peak RAM (MB) during a single forward pass."""
    model = model.to(device).eval()
    dummy = torch.randn(*input_size, device=device)

    tracemalloc.start()
    try:
        with torch.no_grad():
            model(dummy)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    # If psutil available, also capture process RSS
    rss_mb = 0.0
    if _HAS_PSUTIL:
        try:
            rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            rss_mb = 0.0

    return {
        "peak_ram_mb": round(float(peak_mb), 4),
        "peak_memory_mb": round(float(peak_mb), 4),
        "ram_mb": round(float(peak_mb), 4),
        "memory_mb": round(float(peak_mb), 4),
        "rss_mb": round(float(rss_mb), 4),
    }


def profile_model(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
    n_warmup: int = 3,
    n_runs: int = 10,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Full edge-device profile for a single model."""
    params = count_parameters(model)
    flops = estimate_flops(model, input_size=input_size, device=device)
    latency = measure_latency(model, input_size=input_size, device=device, n_warmup=n_warmup, n_runs=n_runs)
    ram = measure_peak_ram(model, input_size=input_size, device=device)

    h, w = input_size[-2], input_size[-1]
    result = {
        "input_size": list(input_size),
        "device": device,
        "params": params["total_params"],
        "param_count": params["total_params"],
        "parameters": params["total_params"],
        "n_params": params["total_params"],
        "params_m": round(params["total_params"] / 1e6, 4),
        "params_millions": round(params["total_params"] / 1e6, 4),
        "flops": flops,
        "flops_g": round(flops / 1e9, 4),
        "gflops": round(flops / 1e9, 4),
        "g_flops": round(flops / 1e9, 4),
    }
    result.update(latency)
    result.update(ram)
    result.update(params)
    return result


def benchmark_edge_devices(
    models: Dict[str, nn.Module],
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Profile a dict of models and return a list of result dicts with model name."""
    results = []
    for name, model in models.items():
        prof = profile_model(model, input_size=input_size, device=device, **kwargs)
        prof["model"] = name
        prof["model_name"] = name
        results.append(prof)
    return results


# Aliases
profile = profile_model
benchmark_model = profile_model
benchmark_edge_device = profile_model
edge_profile = profile_model
edge_device_profile = profile_model
profile_edge_device = profile_model
device_profile = profile_model
latency_benchmark = measure_latency
ram_benchmark = measure_peak_ram
flops_benchmark = estimate_flops
benchmark_models = benchmark_edge_devices
edge_benchmark = benchmark_edge_devices
