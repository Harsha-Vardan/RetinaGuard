"""Tests for edge-device profiling."""

import torch

from src.models.retinaguard import RetinaGuardModel
from src.evaluation.edge_profiling import profile_model, benchmark_edge_devices, count_parameters


def test_profile_keys():
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    prof = profile_model(model, input_size=(1, 3, 32, 32), n_runs=2, n_warmup=1)
    for key in ["params_m", "flops_g", "latency_ms", "peak_ram_mb", "fps"]:
        assert key in prof


def test_param_count_positive():
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    cp = count_parameters(model)
    assert cp["total_params"] > 0
    assert cp["params_m"] if "params_m" in cp else True


def test_benchmark_multiple():
    m1 = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    m2 = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    res = benchmark_edge_devices({"a": m1, "b": m2}, input_size=(1, 3, 32, 32), n_runs=1, n_warmup=0)
    assert len(res) == 2
    assert res[0]["model"] == "a"


def test_latency_nonnegative():
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    prof = profile_model(model, input_size=(1, 3, 32, 32), n_runs=2, n_warmup=0)
    assert prof["latency_ms"] >= 0
    assert prof["peak_ram_mb"] >= 0
