"""RetinaGuard - Unified Lesion-Guided, Causally Informed Model.

Integrates:
1. Spatial Lesion Attention: Guides representations toward clinically relevant lesions.
2. Causal Disentanglement: Isolates invariant disease features from domain/hospital shortcuts.
3. Domain Invariance via GRL: Prevents device-specific shortcuts from affecting DR grading.
4. Statistical Independence Penalty: Minimizes cross-covariance between subspaces.

Usage:
    from src.models.retinaguard import RetinaGuardModel
    model = RetinaGuardModel(num_classes=5, num_domains=2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.lesion_guided import SpatialLesionAttention, compute_lesion_loss
from src.models.causal_disentangler import grad_reverse, compute_covariance_penalty


class RetinaGuardModel(nn.Module):
    """Complete RetinaGuard model unifying lesion guidance and causal disentanglement."""

    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 5,
        num_domains: int = 2,
        latent_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.latent_dim = latent_dim

        # Backbone
        if architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            in_features = 512
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Lesion spatial attention module
        self.lesion_attention = SpatialLesionAttention(in_channels=in_features)

        # Disentanglement projections: operate on fused (global + lesion-attended) features (1024-dim)
        fused_dim = in_features * 2

        self.proj_disease = nn.Sequential(
            nn.Linear(fused_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.proj_domain = nn.Sequential(
            nn.Linear(fused_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.disease_classifier = nn.Linear(latent_dim, num_classes)
        self.domain_classifier = nn.Linear(latent_dim, num_domains)
        self.domain_adversary = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_domains),
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
        self, x: torch.Tensor, alpha: float = 1.0
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input fundus image tensor (B, 3, H, W).
            alpha: Adversarial gradient scaling factor.

        Returns:
            Dictionary of outputs:
              - disease_logits: Logits for DR grade (B, num_classes)
              - domain_logits: Logits for domain from Z_domain (B, num_domains)
              - adv_domain_logits: Logits from GRL on Z_disease (B, num_domains)
              - attn_logits: Raw lesion attention map logits (B, 1, H', W')
              - attn_mask: Sigmoid attention mask (B, 1, H', W')
              - z_disease: Disease-causal latent vector (B, latent_dim)
              - z_domain: Domain shortcut latent vector (B, latent_dim)
        """
        feat_map = self.extract_features(x)  # (B, 512, H', W')

        # Global average pool
        global_feat = torch.mean(feat_map, dim=(2, 3))  # (B, 512)

        # Lesion attention pool
        attn_logits, attn_mask = self.lesion_attention(feat_map)
        weights = attn_mask / (torch.sum(attn_mask, dim=(2, 3), keepdim=True) + 1e-6)
        lesion_feat = torch.sum(feat_map * weights, dim=(2, 3))  # (B, 512)

        # Fused pathology-aware vector
        fused = torch.cat([global_feat, lesion_feat], dim=1)  # (B, 1024)

        # Causal disentanglement
        z_disease = self.proj_disease(fused)  # (B, latent_dim)
        z_domain = self.proj_domain(fused)    # (B, latent_dim)

        # Predictions
        disease_logits = self.disease_classifier(z_disease)
        domain_logits = self.domain_classifier(z_domain)

        # Domain adversarial branch with GRL
        z_disease_rev = grad_reverse(z_disease, alpha)
        adv_domain_logits = self.domain_adversary(z_disease_rev)

        return {
            "disease_logits": disease_logits,
            "domain_logits": domain_logits,
            "adv_domain_logits": adv_domain_logits,
            "attn_logits": attn_logits,
            "attn_mask": attn_mask,
            "z_disease": z_disease,
            "z_domain": z_domain,
        }

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract robust, invariant disease representation."""
        feat_map = self.extract_features(x)
        global_feat = torch.mean(feat_map, dim=(2, 3))
        _, attn_mask = self.lesion_attention(feat_map)
        weights = attn_mask / (torch.sum(attn_mask, dim=(2, 3), keepdim=True) + 1e-6)
        lesion_feat = torch.sum(feat_map * weights, dim=(2, 3))
        fused = torch.cat([global_feat, lesion_feat], dim=1)
        return self.proj_disease(fused)
