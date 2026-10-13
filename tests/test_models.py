"""RetinaGuard - Unit tests for model architecture."""

import torch
import pytest

from src.models.baseline import DRClassifier


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
