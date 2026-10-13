"""RetinaGuard - Causally Informed Disentanglement Framework.

Implements structural causal disentanglement to separate invariant disease
features (Z_disease) from domain- and acquisition-specific shortcuts (Z_domain)
such as camera model, hospital site, illumination, and color variations.

Usage:
    from src.models.causal_disentangler import CausalDisentangler
    model = CausalDisentangler(num_classes=5, num_domains=2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GradientReversalFunction(torch.autograd.Function):
    """Reverses gradient signs during backward pass for domain adversarial training."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal."""
    return GradientReversalFunction.apply(x, alpha)


class CausalDisentangler(nn.Module):
    """Disentangles representations into disease-causal and domain-shortcut subspaces.

    Architecture:
      Backbone -> Shared Embedding Z ->
        Z_disease (invariant pathology features) -> Disease Classifier
                                                 -> GRL -> Domain Adversary
        Z_domain (spurious shortcut features)   -> Domain Classifier
    """

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

        # 1. Feature extractor backbone
        if architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            num_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Architecture {architecture} not supported.")

        # 2. Projection heads for disentangled subspaces
        self.proj_disease = nn.Sequential(
            nn.Linear(num_features, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.proj_domain = nn.Sequential(
            nn.Linear(num_features, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 3. Task Classifiers
        # Disease grade predictor operates ONLY on Z_disease
        self.disease_classifier = nn.Linear(latent_dim, num_classes)

        # Domain predictor operates on Z_domain (learns hospital/device shortcuts)
        self.domain_classifier = nn.Linear(latent_dim, num_domains)

        # Domain adversary operates on Z_disease via GRL (ensures Z_disease is domain-invariant)
        self.domain_adversary = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_domains),
        )

    def forward(
        self, x: torch.Tensor, alpha: float = 1.0
    ) -> dict[str, torch.Tensor]:
        """Forward pass through causal disentangler.

        Args:
            x: Input fundus image tensor (B, 3, H, W).
            alpha: Adversarial gradient scaling factor.

        Returns:
            Dictionary containing:
              - disease_logits: Logits for DR grading (B, num_classes)
              - domain_logits: Logits for domain classification from Z_domain (B, num_domains)
              - adv_domain_logits: Logits for domain classification from Z_disease with GRL (B, num_domains)
              - z_disease: Disease feature representations (B, latent_dim)
              - z_domain: Domain shortcut representations (B, latent_dim)
        """
        shared_feat = self.backbone(x)  # (B, 512)

        # Project into orthogonal representations
        z_disease = self.proj_disease(shared_feat)  # (B, latent_dim)
        z_domain = self.proj_domain(shared_feat)    # (B, latent_dim)

        # Predictions
        disease_logits = self.disease_classifier(z_disease)
        domain_logits = self.domain_classifier(z_domain)

        # Adversarial domain prediction with GRL
        z_disease_rev = grad_reverse(z_disease, alpha)
        adv_domain_logits = self.domain_adversary(z_disease_rev)

        return {
            "disease_logits": disease_logits,
            "domain_logits": domain_logits,
            "adv_domain_logits": adv_domain_logits,
            "z_disease": z_disease,
            "z_domain": z_domain,
        }

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract purely disease-causal features for robust inference."""
        shared_feat = self.backbone(x)
        return self.proj_disease(shared_feat)


def compute_covariance_penalty(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Compute Frobenius norm of cross-covariance to enforce statistical independence.

    Minimizing ||Cov(Z_disease, Z_domain)||_F^2 guarantees that disease
    representations are uncorrelated with hospital/imaging shortcuts.

    Args:
        z1: Feature tensor 1 (B, D1).
        z2: Feature tensor 2 (B, D2).

    Returns:
        Scalar loss tensor.
    """
    if z1.shape[0] <= 1:
        return torch.tensor(0.0, device=z1.device)

    # Center representations
    z1_centered = z1 - torch.mean(z1, dim=0, keepdim=True)
    z2_centered = z2 - torch.mean(z2, dim=0, keepdim=True)

    # Cross-covariance matrix
    cov = torch.mm(z1_centered.t(), z2_centered) / (z1.shape[0] - 1)

    # Mean squared cross-covariance (calibrated scale)
    return torch.mean(cov ** 2)
