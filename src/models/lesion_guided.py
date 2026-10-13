"""RetinaGuard - Lesion-Guided DR Classification Model.

Implements an attention-guided architecture that identifies pathology-bearing
regions (microaneurysms, hemorrhages, exudates) to guide classification decisions
and prevent spurious shortcut learning.

Usage:
    from src.models.lesion_guided import LesionGuidedClassifier
    model = LesionGuidedClassifier(architecture="resnet18", num_classes=5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialLesionAttention(nn.Module):
    """Generates spatial attention weights emphasizing lesion areas."""

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(16, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate spatial lesion attention mask.

        Args:
            x: Feature map (B, C, H, W).

        Returns:
            tuple of (attention_logits, attention_mask) each shape (B, 1, H, W).
        """
        feat = F.relu(self.bn1(self.conv1(x)))
        attn_logits = self.conv2(feat)
        attn_mask = torch.sigmoid(attn_logits)
        return attn_logits, attn_mask


class LesionGuidedClassifier(nn.Module):
    """Lesion-guided retinal fundus classification model.

    Combines whole-image representation with lesion-attended local features
    and supports auxiliary supervision using ground-truth lesion segmentation masks.
    """

    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes

        if architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            # Feature extractor up to layer4
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            in_features = 512
        else:
            raise ValueError(f"Architecture {architecture} not supported for lesion guidance yet.")

        # Spatial lesion attention head on layer4 features
        self.lesion_attention = SpatialLesionAttention(in_channels=in_features)

        # Classification head combining global and lesion-guided pooled features
        # [global_pool (512) + lesion_pool (512)] -> 1024
        self.classifier = nn.Sequential(
            nn.Linear(in_features * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input image tensor (B, 3, H, W).

        Returns:
            tuple of:
              - logits: DR classification predictions (B, num_classes).
              - attn_logits: Raw attention map logits (B, 1, H', W').
              - attn_mask: Sigmoid attention mask (B, 1, H', W').
        """
        feat_map = self.extract_features(x)  # (B, 512, H', W')

        # 1. Global average pooling
        global_feat = torch.mean(feat_map, dim=(2, 3))  # (B, 512)

        # 2. Lesion spatial attention
        attn_logits, attn_mask = self.lesion_attention(feat_map)  # (B, 1, H', W')

        # 3. Attention-weighted pooling
        # Weighted average over spatial dimensions
        weights = attn_mask / (torch.sum(attn_mask, dim=(2, 3), keepdim=True) + 1e-6)
        lesion_feat = torch.sum(feat_map * weights, dim=(2, 3))  # (B, 512)

        # 4. Fused representation
        fused = torch.cat([global_feat, lesion_feat], dim=1)  # (B, 1024)
        logits = self.classifier(fused)

        return logits, attn_logits, attn_mask

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return fused representation vector for downstream evaluation."""
        feat_map = self.extract_features(x)
        global_feat = torch.mean(feat_map, dim=(2, 3))
        _, attn_mask = self.lesion_attention(feat_map)
        weights = attn_mask / (torch.sum(attn_mask, dim=(2, 3), keepdim=True) + 1e-6)
        lesion_feat = torch.sum(feat_map * weights, dim=(2, 3))
        return torch.cat([global_feat, lesion_feat], dim=1)


def compute_lesion_loss(
    attn_logits: torch.Tensor,
    gt_masks: torch.Tensor | None,
    mask_available: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute auxiliary lesion attention supervision loss.

    Args:
        attn_logits: Predicted attention map logits (B, 1, H', W').
        gt_masks: Ground truth lesion binary masks (B, 1, H, W).
        mask_available: 1D boolean tensor of shape (B,) indicating which
                        samples have real lesion masks.

    Returns:
        Scalar loss tensor. Returns 0 if no masks available.
    """
    if gt_masks is None or mask_available is None or not mask_available.any():
        return torch.tensor(0.0, device=attn_logits.device, requires_grad=True)

    # Filter to samples that have real masks
    sub_logits = attn_logits[mask_available]
    sub_gt = gt_masks[mask_available]

    # Downsample ground truth mask to match attention map spatial resolution
    target_masks = F.interpolate(
        sub_gt.float(), size=sub_logits.shape[-2:], mode="area"
    )
    # Binarize downsampled target
    target_masks = (target_masks > 0.05).float()

    # Binary cross entropy with logits
    loss = F.binary_cross_entropy_with_logits(sub_logits, target_masks)
    return loss
