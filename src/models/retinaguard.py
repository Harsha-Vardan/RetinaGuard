"""RetinaGuard - Unified Lesion-Guided, Causally Informed Model.

Integrates:
1. Spatial Lesion Attention: Guides representations toward clinically relevant lesions.
2. Causal Disentanglement: Isolates invariant disease features from domain/hospital shortcuts.
3. Domain Invariance via GRL: Prevents device-specific shortcuts from affecting DR grading.
4. Statistical Independence Penalty: Minimizes cross-covariance between subspaces.

Usage:
    from src.models.retinaguard import RetinaGuardModel

    # 1. Initialize from scratch
    model = RetinaGuardModel(num_classes=5, num_domains=2)

    # 2. Auto-load the best trained medical checkpoint from experiments/
    model = RetinaGuardModel.from_trained()

    # 3. Load specific trained checkpoint
    model = RetinaGuardModel.from_trained("experiments/retinaguard/best_retinaguard.pth")
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torchvision.models as models

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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
        checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.latent_dim = latent_dim

        self.architecture = architecture
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
        elif architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            in_features = 2048
        elif architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            self.backbone_features = base.features
            in_features = 1280
        elif architecture == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = models.convnext_tiny(weights=weights)
            self.backbone_features = base.features
            in_features = 768
        else:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                "Supported: ['resnet18', 'resnet50', 'efficientnet_b0', 'convnext_tiny']"
            )

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

        # Classification heads
        self.disease_classifier = nn.Linear(latent_dim, num_classes)
        self.domain_classifier = nn.Linear(latent_dim, num_domains)
        self.domain_adversary = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_domains),
        )

        # Optional loading of trained weights
        if checkpoint_path is not None:
            ckpt_p = Path(checkpoint_path)
            if ckpt_p.exists():
                data = torch.load(ckpt_p, map_location="cpu", weights_only=False)
                state_dict = data.get("model_state_dict", data)
                try:
                    self.load_state_dict(state_dict)
                except Exception:
                    self.load_state_dict(state_dict, strict=False)

    @classmethod
    def from_trained(
        cls,
        checkpoint_path: Optional[Union[str, Path]] = None,
        architecture: str = "resnet18",
        num_classes: int = 5,
        num_domains: int = 2,
        latent_dim: int = 256,
        device: Union[str, torch.device] = "cpu",
    ) -> "RetinaGuardModel":
        """Load an already trained RetinaGuardModel, auto-discovering best checkpoint if not specified.

        Args:
            checkpoint_path: Path to checkpoint, or None to auto-find best_retinaguard.pth.
            architecture: Model architecture.
            num_classes: Number of DR classes (default 5).
            num_domains: Number of training domains (default 2).
            latent_dim: Latent representation size (default 256).
            device: Target device (cpu or cuda).

        Returns:
            Trained RetinaGuardModel in evaluation mode.
        """
        if checkpoint_path is None:
            candidates = sorted(
                list(PROJECT_ROOT.glob("experiments/**/best_retinaguard.pth")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                raise FileNotFoundError(
                    "No trained RetinaGuard checkpoint found in experiments/. "
                    "Train RetinaGuard first via `python -m src.training.train_retinaguard`."
                )
            checkpoint_path = candidates[0]

        model = cls(
            architecture=architecture,
            num_classes=num_classes,
            num_domains=num_domains,
            latent_dim=latent_dim,
            pretrained=False,
        )
        data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = data.get("model_state_dict", data)
        try:
            model.load_state_dict(state_dict)
        except Exception:
            model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        return model

    @classmethod
    def load_trained(
        cls,
        checkpoint_path: Optional[Union[str, Path]] = None,
        architecture: str = "resnet18",
        num_classes: int = 5,
        num_domains: int = 2,
        latent_dim: int = 256,
        device: Union[str, torch.device] = "cpu",
    ) -> "RetinaGuardModel":
        """Alias for from_trained()."""
        return cls.from_trained(
            checkpoint_path=checkpoint_path,
            architecture=architecture,
            num_classes=num_classes,
            num_domains=num_domains,
            latent_dim=latent_dim,
            device=device,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale convolutional feature maps."""
        if hasattr(self, "backbone_features"):
            return self.backbone_features(x)
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
