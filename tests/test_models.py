"""RetinaGuard - Unit tests for model architecture."""

from pathlib import Path
import pytest
import torch

from src.models.baseline import DRClassifier
from src.models.retinaguard import RetinaGuardModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_resnet18_forward_pass():
    """Verify ResNet18 forward pass shape and output dimensions."""
    model = DRClassifier(architecture="resnet18", num_classes=5, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert out.shape == (2, 5)
    assert not torch.isnan(out).any()


def test_feature_extraction():
    """Verify feature extractor returns backbone representations."""
    model = DRClassifier(architecture="resnet18", num_classes=5, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    features = model.get_features(x)

    # ResNet18 avgpool output dimension is 512
    assert features.shape == (2, 512)


def test_freeze_backbone():
    """Verify backbone parameter freezing."""
    model = DRClassifier(
        architecture="resnet18", num_classes=5, pretrained=False, freeze_backbone=True
    )
    info = model.get_model_info()

    assert info["frozen_parameters"] > 0
    # Classifier head should still be trainable
    assert info["trainable_parameters"] > 0
    assert info["trainable_parameters"] < info["total_parameters"]


def test_supported_architectures():
    """Verify initialization across supported architectures without error."""
    for arch in ["resnet18", "efficientnet_b0", "mobilenet_v3_small"]:
        model = DRClassifier(architecture=arch, num_classes=5, pretrained=False)
        info = model.get_model_info()
        assert info["architecture"] == arch
        assert info["total_parameters"] > 0


def test_unsupported_architecture():
    """Verify ValueError on unsupported architecture."""
    with pytest.raises(ValueError, match="Unsupported architecture"):
        DRClassifier(architecture="nonexistent_net", pretrained=False)


def test_dr_classifier_from_trained_auto_discovery():
    """Verify that DRClassifier.from_trained() loads trained weights if present."""
    ckpt_path = PROJECT_ROOT / "experiments" / "baseline" / "best_model.pth"
    if not ckpt_path.exists():
        pytest.skip("No trained baseline checkpoint on disk")

    model = DRClassifier.from_trained()
    assert isinstance(model, DRClassifier)
    assert not model.training  # In eval mode

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 5)


def test_retinaguard_from_trained_auto_discovery():
    """Verify that RetinaGuardModel.from_trained() loads trained multi-task weights."""
    ckpt_path = PROJECT_ROOT / "experiments" / "retinaguard" / "best_retinaguard.pth"
    if not ckpt_path.exists():
        pytest.skip("No trained RetinaGuard checkpoint on disk")

    model = RetinaGuardModel.from_trained()
    assert isinstance(model, RetinaGuardModel)
    assert not model.training

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert "disease_logits" in out
    assert "attn_mask" in out
    assert out["disease_logits"].shape == (1, 5)
