"""RetinaGuard - Attribution Faithfulness and Sanity Checks.

Implements sanity checks recommended by Adebayo et al. (NeurIPS 2018):
1. Model Parameter Randomization Test: Feature attribution must degrade/diverge when
   network weights are randomized, proving explanations are model-dependent.
2. Pointing Game hit boundary behavior.
3. Attention map sparsity and bounds on input images.
"""

import numpy as np
import pytest
import torch

from src.explainability.attribution import (
    compute_attribution_iou,
    compute_pointing_game_hit,
    GradCAM,
)
from src.models.baseline import DRClassifier
from src.models.retinaguard import RetinaGuardModel


def test_adebayo_parameter_randomization_sanity():
    """Verify that randomizing model weights disrupts the feature attribution map.

    Adebayo et al. (Sanity Checks for Saliency Maps): If a saliency method is faithful
    to the learned parameters, re-initializing weights from top down must radically
    alter the attribution heatmap.
    """
    torch.manual_seed(42)
    model = DRClassifier(architecture="resnet18", num_classes=5, pretrained=False)
    model.eval()

    img = torch.randn(1, 3, 128, 128, requires_grad=True)

    # Compute Grad-CAM with current weights
    grad_cam = GradCAM(model, model.backbone.layer4)
    cam_orig = grad_cam.generate(img, target_class=2)

    # Randomize final layer weights
    with torch.no_grad():
        model.classifier[1].weight.copy_(torch.randn_like(model.classifier[1].weight) * 2.0)
        model.classifier[1].bias.copy_(torch.randn_like(model.classifier[1].bias))

    cam_randomized = grad_cam.generate(img, target_class=2)

    # Check that the two heatmaps are not identical (weight change alters explanation)
    diff = np.abs(cam_orig - cam_randomized)
    assert np.max(diff) > 0.05
    assert not np.allclose(cam_orig, cam_randomized, atol=1e-3)


def test_pointing_game_boundary_coordinates():
    """Verify pointing game hit calculation at exact image boundary pixels."""
    # 10x10 heatmap with peak at (0, 0)
    heatmap = np.zeros((10, 10), dtype=float)
    heatmap[0, 0] = 1.0

    # Mask with lesion at (0, 0)
    mask_hit = np.zeros((10, 10), dtype=np.uint8)
    mask_hit[0, 0] = 255
    assert compute_pointing_game_hit(heatmap, mask_hit, tolerance_radius=0) == 1.0

    # Mask with lesion elsewhere
    mask_miss = np.zeros((10, 10), dtype=np.uint8)
    mask_miss[9, 9] = 255
    assert compute_pointing_game_hit(heatmap, mask_miss, tolerance_radius=0) == 0.0


def test_pointing_game_all_zeros_mask():
    """Verify pointing game returns 0.0 when mask has no lesions and model attends strongly."""
    heatmap = np.ones((20, 20)) * 0.90  # Strong activation everywhere
    empty_mask = np.zeros((20, 20), dtype=np.uint8)
    assert compute_pointing_game_hit(heatmap, empty_mask) == 0.0


def test_attribution_iou_monotonicity():
    """Verify attribution IoU increases monotonically with expanding mask overlap."""
    heatmap = np.zeros((50, 50), dtype=float)
    heatmap[10:30, 10:30] = 0.90  # 20x20 active block

    # GT mask 1: 10x10 overlap
    mask1 = np.zeros((50, 50), dtype=np.uint8)
    mask1[10:20, 10:20] = 255
    iou1 = compute_attribution_iou(heatmap, mask1, threshold=0.5)

    # GT mask 2: 20x20 exact overlap
    mask2 = np.zeros((50, 50), dtype=np.uint8)
    mask2[10:30, 10:30] = 255
    iou2 = compute_attribution_iou(heatmap, mask2, threshold=0.5)

    assert iou2 > iou1
    assert np.isclose(iou2, 1.0, atol=1e-3)


def test_lesion_attention_entropy():
    """Verify lesion attention head produces bounded spatial attention weights."""
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    model.eval()

    img = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(img)
        attn = out["attn_mask"]  # shape (2, 1, H', W')

    assert attn.min() >= 0.0
    assert attn.max() <= 1.0
    assert attn.shape[0] == 2
    assert attn.shape[1] == 1
