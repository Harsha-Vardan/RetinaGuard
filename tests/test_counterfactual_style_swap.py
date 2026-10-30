"""Tests for causal counterfactual style swapping."""

import torch

from src.models.retinaguard import RetinaGuardModel
from src.evaluation.counterfactual import (
    counterfactual_style_swap,
    counterfactual_invariance_rate,
    extract_latents,
)


def test_counterfactual_swap_invariant():
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    model.eval()
    xa = torch.randn(1, 3, 64, 64)
    xb = torch.randn(1, 3, 64, 64)
    res = counterfactual_style_swap(model, xa, xb)
    assert "invariant" in res
    assert "predicted_original" in res
    assert "predicted_counterfactual" in res
    assert res["predicted_original"] == res["predicted_counterfactual"]
    assert res["invariant"] is True


def test_extract_latents_shapes():
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False, latent_dim=32)
    x = torch.randn(2, 3, 64, 64)
    lat = extract_latents(model, x)
    assert lat["z_disease"].shape == (2, 32)
    assert lat["z_domain"].shape == (2, 32)


def test_invariance_rate_bounds():
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    imgs = torch.randn(4, 3, 64, 64)
    rate = counterfactual_invariance_rate(model, imgs)
    assert 0.0 <= rate <= 1.0


def test_counterfactual_batch_shape():
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    xa = torch.randn(3, 64, 64).unsqueeze(0)
    xb = torch.randn(1, 3, 64, 64)
    res = counterfactual_style_swap(model, xa, xb)
    assert res["z_disease_a"].shape[0] == 1
