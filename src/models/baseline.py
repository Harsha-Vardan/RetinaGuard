"""RetinaGuard - Baseline DR Classification Model.

Lightweight pretrained model for five-class diabetic retinopathy grading.
Default: ResNet18 with ImageNet pretrained weights.

Usage:
    from src.models.baseline import DRClassifier
    model = DRClassifier(architecture="resnet18", num_classes=5, pretrained=True)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DRClassifier(nn.Module):
    """Baseline DR classification model using pretrained backbones.

    Supports ResNet18, EfficientNet-B0, and MobileNetV3-Small.
    Uses transfer learning with a custom classification head.
    """

    SUPPORTED_ARCHITECTURES = ["resnet18", "efficientnet_b0", "mobilenet_v3_small"]

    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        """Initialize the DR classifier.

        Args:
            architecture: Backbone architecture name.
            num_classes: Number of output classes (5 for DR grading).
            pretrained: Whether to use ImageNet pretrained weights.
            dropout: Dropout rate before final classifier.
            freeze_backbone: Whether to freeze backbone parameters.
        """
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes

        if architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes),
            )

        elif architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes),
            )

        elif architecture == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes),
            )

        else:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Choose from: {self.SUPPORTED_ARCHITECTURES}"
            )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.pretrained_source = (
            f"torchvision_{architecture}_imagenet1k" if pretrained else "random"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, H, W).

        Returns:
            Logits of shape (batch, num_classes).
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone without classification.

        Args:
            x: Input tensor of shape (batch, 3, H, W).

        Returns:
            Feature tensor from the backbone.
        """
        return self.backbone(x)

    def get_model_info(self) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "pretrained_source": self.pretrained_source,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
        }
