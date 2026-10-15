"""RetinaGuard - Tests for Causal Feature Disentanglement and Invariance."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.causal_disentangler import (
    CausalDisentangler,
    compute_covariance_penalty,
    grad_reverse,
)
from src.models.retinaguard import RetinaGuardModel


def test_grl_gradient_inversion_exact():
    """Verify that Gradient Reversal Layer scales gradient by exactly -alpha."""
    x = torch.tensor([2.0, -3.0], requires_grad=True)
    alpha = 0.75
    y = grad_reverse(x, alpha=alpha)

    # Forward pass is identity
    assert torch.all(x == y)

    # Backward pass multiplies gradient by -alpha
    loss = (y * torch.tensor([1.0, 2.0])).sum()
    loss.backward()

    expected_grad = -alpha * torch.tensor([1.0, 2.0])
    assert torch.allclose(x.grad, expected_grad, atol=1e-6)


def test_covariance_penalty_zero_correlation():
    """Verify covariance penalty is zero when disease and artifact latents are uncorrelated across batch."""
    # 4 samples with balanced orthogonal variations such that cov matrix is zero
    z_d = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
    z_a = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, -1.0], [0.0, -1.0]], dtype=torch.float32)

    cov_pen = compute_covariance_penalty(z_d, z_a)
    assert np.isclose(cov_pen.item(), 0.0, atol=1e-6)


def test_covariance_penalty_collinear_subspaces():
    """Verify covariance penalty is strictly positive when disease and artifact latents are correlated."""
    z_d = torch.tensor([[1.0, 2.0], [-1.0, -2.0]], dtype=torch.float32)
    z_a = torch.tensor([[2.0, 4.0], [-2.0, -4.0]], dtype=torch.float32)

    cov_pen = compute_covariance_penalty(z_d, z_a)
    assert cov_pen.item() > 0.0


def test_causal_disentangler_forward_shapes():
    """Verify CausalDisentangler outputs expected latent and logit shapes."""
    disentangler = CausalDisentangler(
        architecture="resnet18",
        latent_dim=128,
        num_classes=5,
        num_domains=2,
        pretrained=False,
    )
    images = torch.randn(4, 3, 128, 128)

    out = disentangler(images, alpha=0.5)

    assert out["z_disease"].shape == (4, 128)
    assert out["z_domain"].shape == (4, 128)
    assert out["disease_logits"].shape == (4, 5)
    assert out["domain_logits"].shape == (4, 2)
    assert out["adv_domain_logits"].shape == (4, 2)


def test_retinaguard_disentanglement_training_step():
    """Verify full RetinaGuard model computes non-zero gradients for all heads in multi-task step."""
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    model.train()

    images = torch.randn(2, 3, 128, 128)
    out = model(images, alpha=0.5)

    ce_loss = nn.CrossEntropyLoss()(out["disease_logits"], torch.tensor([0, 2]))
    dom_loss = nn.CrossEntropyLoss()(out["domain_logits"], torch.tensor([0, 1]))
    cov_loss = compute_covariance_penalty(out["z_disease"], out["z_domain"])

    total_loss = ce_loss + 0.1 * dom_loss + 0.05 * cov_loss
    total_loss.backward()

    # Verify gradients reach backbone stem and projection heads
    assert model.stem[0].weight.grad is not None
    assert model.proj_disease[0].weight.grad is not None
    assert model.proj_domain[0].weight.grad is not None
