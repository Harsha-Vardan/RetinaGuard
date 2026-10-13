"""RetinaGuard - Unit tests for research models (Lesion-Guided & Causal Disentanglement)."""

import torch
import torch.nn.functional as F
import pytest

from src.models.lesion_guided import (
    SpatialLesionAttention,
    LesionGuidedClassifier,
    compute_lesion_loss,
)
from src.models.causal_disentangler import (
    CausalDisentangler,
    grad_reverse,
    compute_covariance_penalty,
)
from src.models.retinaguard import RetinaGuardModel


def test_spatial_lesion_attention():
    """Verify spatial attention outputs valid probability mask."""
    attn_module = SpatialLesionAttention(in_channels=64)
    x = torch.randn(2, 64, 16, 16)
    logits, mask = attn_module(x)

    assert logits.shape == (2, 1, 16, 16)
    assert mask.shape == (2, 1, 16, 16)
    assert (mask >= 0.0).all() and (mask <= 1.0).all()


def test_lesion_guided_classifier():
    """Verify lesion-guided classifier forward pass and shapes."""
    model = LesionGuidedClassifier(architecture="resnet18", num_classes=5, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits, attn_logits, attn_mask = model(x)

    assert logits.shape == (2, 5)
    assert attn_mask.shape[0] == 2
    assert attn_mask.shape[1] == 1

    features = model.get_features(x)
    assert features.shape == (2, 1024)


def test_lesion_loss_computation():
    """Verify auxiliary lesion loss calculates BCE on masked samples."""
    attn_logits = torch.randn(2, 1, 14, 14, requires_grad=True)
    gt_masks = torch.ones(2, 1, 224, 224)
    mask_available = torch.tensor([True, False])

    loss = compute_lesion_loss(attn_logits, gt_masks, mask_available)
    assert loss.item() > 0
    loss.backward()
    assert attn_logits.grad is not None


def test_gradient_reversal_layer():
    """Verify GRL reverses gradients during backward pass."""
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    alpha = 2.5
    y = grad_reverse(x, alpha)
    loss = (y * 3).sum()
    loss.backward()

    # dLoss/dx should be 3 * -alpha = -7.5
    assert torch.allclose(x.grad, torch.tensor([-7.5, -7.5]))


def test_covariance_penalty():
    """Verify covariance penalty is zero for identical constants and non-zero for correlated vectors."""
    z1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    z2 = torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
    cov_loss = compute_covariance_penalty(z1, z2)
    assert cov_loss.item() > 0


def test_causal_disentangler_forward():
    """Verify causal disentangler output dictionary and dimensions."""
    model = CausalDisentangler(
        architecture="resnet18", num_classes=5, num_domains=2, latent_dim=128, pretrained=False
    )
    x = torch.randn(3, 3, 224, 224)
    out = model(x, alpha=1.0)

    assert "disease_logits" in out
    assert "domain_logits" in out
    assert "adv_domain_logits" in out
    assert "z_disease" in out
    assert "z_domain" in out

    assert out["disease_logits"].shape == (3, 5)
    assert out["domain_logits"].shape == (3, 2)
    assert out["adv_domain_logits"].shape == (3, 2)
    assert out["z_disease"].shape == (3, 128)
    assert out["z_domain"].shape == (3, 128)


def test_unified_retinaguard_model():
    """Verify unified RetinaGuard model forward and backward pass."""
    model = RetinaGuardModel(
        architecture="resnet18", num_classes=5, num_domains=2, latent_dim=128, pretrained=False
    )
    x = torch.randn(2, 3, 224, 224)
    out = model(x, alpha=1.0)

    assert out["disease_logits"].shape == (2, 5)
    assert out["attn_mask"].shape[1] == 1
    assert out["z_disease"].shape == (2, 128)

    # Multi-task loss
    loss = (
        F.cross_entropy(out["disease_logits"], torch.tensor([0, 2]))
        + F.cross_entropy(out["domain_logits"], torch.tensor([0, 1]))
        + F.cross_entropy(out["adv_domain_logits"], torch.tensor([0, 1]))
        + compute_covariance_penalty(out["z_disease"], out["z_domain"])
    )
    loss.backward()

    # Check gradients exist
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()
