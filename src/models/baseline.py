"""RetinaGuard - Baseline DR Classification Model.

Lightweight pretrained model for five-class diabetic retinopathy grading.
Default: ResNet18 with ImageNet pretrained weights or pre-trained RetinaGuard checkpoints.

Usage:
    from src.models.baseline import DRClassifier

    # 1. Initialize with ImageNet weights
    model = DRClassifier(architecture="resnet18", num_classes=5, pretrained=True)

    # 2. Auto-load the best trained medical checkpoint from experiments/
    model = DRClassifier.from_trained()

    # 3. Load specific trained checkpoint
    model = DRClassifier.from_trained("experiments/baseline/best_model.pth")
"""

import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torchvision.models as models

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DRClassifier(nn.Module):
    """Baseline DR classification model using pretrained backbones.

    Supports ResNet18, EfficientNet-B0, and MobileNetV3-Small.
    Uses transfer learning with a custom classification head.
    Supports auto-loading already trained medical checkpoints.
    """

    SUPPORTED_ARCHITECTURES = ["resnet18", "efficientnet_b0", "mobilenet_v3_small"]

    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize the DR classifier.

        Args:
            architecture: Backbone architecture name.
            num_classes: Number of output classes (5 for DR grading).
            pretrained: Whether to use ImageNet pretrained weights.
            dropout: Dropout rate before final classifier.
            freeze_backbone: Whether to freeze backbone parameters.
            checkpoint_path: Optional path to an already trained checkpoint (.pth) to load.
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

        # Load trained weights if checkpoint specified
        if checkpoint_path is not None:
            ckpt_p = Path(checkpoint_path)
            if ckpt_p.exists():
                data = torch.load(ckpt_p, map_location="cpu", weights_only=False)
                state_dict = data.get("model_state_dict", data)
                self.load_state_dict(state_dict)
                self.pretrained_source = str(ckpt_p)

    @classmethod
    def from_trained(
        cls,
        checkpoint_path: Optional[Union[str, Path]] = None,
        architecture: str = "resnet18",
        num_classes: int = 5,
        device: Union[str, torch.device] = "cpu",
    ) -> "DRClassifier":
        """Load an already trained DRClassifier, auto-discovering the best model if not provided.

        Args:
            checkpoint_path: Path to .pth checkpoint, or None to auto-find best_model.pth.
            architecture: Model architecture.
            num_classes: Number of DR classes.
            device: Target device (cpu or cuda).

        Returns:
            Trained DRClassifier in evaluation mode.
        """
        if checkpoint_path is None:
            candidates = sorted(
                list(PROJECT_ROOT.glob("experiments/**/best_model.pth")) +
                list(PROJECT_ROOT.glob("experiments/**/best_baseline.pth")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                raise FileNotFoundError(
                    "No trained baseline checkpoint found in experiments/. "
                    "Train a baseline first via `python -m src.training.train_baseline`."
                )
            checkpoint_path = candidates[0]

        model = cls(architecture=architecture, num_classes=num_classes, pretrained=False)
        data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = data.get("model_state_dict", data)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        model.pretrained_source = str(checkpoint_path)
        return model

    @classmethod
    def load_trained(
        cls,
        checkpoint_path: Optional[Union[str, Path]] = None,
        architecture: str = "resnet18",
        num_classes: int = 5,
        device: Union[str, torch.device] = "cpu",
    ) -> "DRClassifier":
        """Alias for from_trained()."""
        return cls.from_trained(
            checkpoint_path=checkpoint_path,
            architecture=architecture,
            num_classes=num_classes,
            device=device,
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
