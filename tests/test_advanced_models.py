"""RetinaGuard - Tests for Advanced Causal & Lesion Attention Mechanisms."""

import pytest
import torch
import torch.nn as nn

from src.models.causal_disentangler import CausalDisentangler, GradientReversalFunction, grad_reverse
from src.models.lesion_guided import LesionGuidedClassifier, SpatialLesionAttention
from src.models.retinaguard import RetinaGuardModel, compute_covariance_penalty


def test_gradient_reversal_different_alpha():
    """Verify that GRL scales gradient by exact negative alpha."""
    x = torch.tensor([2.0, -3.0], requires_grad=True)
    out = grad_reverse(x, alpha=0.5)
    loss = (out ** 2).sum()
    loss.backward()

    # Forward pass should be identity: out == x
    assert torch.allclose(out, x)
    # Backward pass: d(x^2)/dx = 2*x. With GRL alpha=0.5, grad = -0.5 * 2 * x = -x
    expected_grad = -x.detach()
    assert torch.allclose(x.grad, expected_grad)


def test_covariance_penalty_properties():
    """Verify covariance penalty is zero for uncorrelated features and high for identical features."""
    # Orthogonal / uncorrelated vectors across 4 samples
    z1 = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
    z2 = torch.tensor([[0.0, 1.0], [0.0, -1.0], [0.0, 1.0], [0.0, -1.0]], dtype=torch.float32)
    cov_zero = compute_covariance_penalty(z1, z2)
    assert cov_zero.item() == pytest.approx(0.0, abs=1e-5)

    # Correlated vectors
    z3 = torch.randn(10, 4)
    cov_high = compute_covariance_penalty(z3, z3)
    assert cov_high.item() > 0.0


def test_spatial_lesion_attention_gradients():
    """Verify that gradients propagate cleanly through spatial lesion attention."""
    attn = SpatialLesionAttention(in_channels=64, reduction=4)
    feat = torch.randn(2, 64, 16, 16, requires_grad=True)

    logits, mask = attn(feat)
    loss = mask.sum()
    loss.backward()

    assert feat.grad is not None
    assert not torch.isnan(feat.grad).any()
    assert (mask >= 0.0).all() and (mask <= 1.0).all()


def test_retinaguard_multi_task_forward():
    """Verify that RetinaGuard computes disease logits, domain logits, and attention mask in one pass."""
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert "disease_logits" in out
    assert "domain_logits" in out
    assert "adv_domain_logits" in out
    assert "z_disease" in out
    assert "z_domain" in out
    assert "attn_mask" in out

    assert out["disease_logits"].shape == (2, 5)
    assert out["domain_logits"].shape == (2, 2)
    assert out["adv_domain_logits"].shape == (2, 2)
    assert out["attn_mask"].shape[0] == 2
    assert out["attn_mask"].shape[1] == 1


def test_causal_disentangler_subspace_separation():
    """Verify that disease classifier and domain classifier have independent weights."""
    model = CausalDisentangler(architecture="resnet18", num_classes=5, num_domains=2, latent_dim=128, pretrained=False)

    assert model.disease_classifier.weight.shape == (5, 128)
    assert model.domain_classifier.weight.shape == (2, 128)
    # Check that disease and domain classifiers have different weights
    assert not torch.equal(model.proj_disease[0].weight, model.proj_domain[0].weight)
