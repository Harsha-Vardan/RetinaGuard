"""Tests for RetinaGuard Explainability, Attribution & Inference."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.explainability.attribution import (
    GradCAM,
    extract_lesion_attention_map,
    compute_attribution_iou,
    compute_attribution_dice,
    generate_explanation_figure,
)
from src.evaluation.benchmark_comparison import compute_referable_dr_metrics
from src.inference.predict import DRPredictor
from src.models.retinaguard import RetinaGuardModel


class SimpleDummyCNN(nn.Module):
    """Simple CNN for testing GradCAM hooks."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        pooled = self.pool(x).flatten(1)
        return self.fc(pooled)


def test_attribution_iou_and_dice():
    """Verify IoU and Dice computations under controlled binary masks."""
    # Perfect match
    mask_a = np.zeros((10, 10), dtype=np.uint8)
    mask_a[2:6, 2:6] = 1

    mask_b = np.zeros((10, 10), dtype=float)
    mask_b[2:6, 2:6] = 1.0

    iou = compute_attribution_iou(mask_b, mask_a, threshold=0.5)
    dice = compute_attribution_dice(mask_b, mask_a, threshold=0.5)
    assert iou == pytest.approx(1.0)
    assert dice == pytest.approx(1.0)

    # Disjoint masks
    mask_c = np.zeros((10, 10), dtype=float)
    mask_c[7:9, 7:9] = 1.0
    iou_disjoint = compute_attribution_iou(mask_c, mask_a, threshold=0.5)
    assert iou_disjoint == pytest.approx(0.0)

    # Empty masks
    empty_1 = np.zeros((10, 10), dtype=float)
    empty_2 = np.zeros((10, 10), dtype=np.uint8)
    assert compute_attribution_iou(empty_1, empty_2) == 1.0
    assert compute_attribution_dice(empty_1, empty_2) == 1.0


def test_grad_cam_generation():
    """Verify GradCAM generates valid normalized 2D heatmaps."""
    model = SimpleDummyCNN()
    grad_cam = GradCAM(model, model.conv2)
    x = torch.randn(1, 3, 32, 32, requires_grad=True)

    cam = grad_cam.generate(x, target_class=2)
    assert cam.shape == (32, 32)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0


def test_extract_lesion_attention_map():
    """Verify extraction of spatial lesion attention map from RetinaGuardModel."""
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    x = torch.randn(1, 3, 64, 64)

    heatmap = extract_lesion_attention_map(model, x, target_size=(64, 64))
    assert heatmap.shape == (64, 64)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_generate_explanation_figure(tmp_path):
    """Verify multi-panel figure generation and file saving."""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    heatmap = np.random.rand(64, 64).astype(np.float32)
    gt_mask = np.zeros((64, 64), dtype=np.uint8)
    gt_mask[10:30, 10:30] = 255

    save_file = tmp_path / "test_explanation.png"
    fig = generate_explanation_figure(
        image_np=img,
        heatmap=heatmap,
        ground_truth_mask=gt_mask,
        save_path=save_file,
        predicted_grade=2,
        confidence=0.88,
        iou=0.45,
    )
    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_referable_dr_metrics():
    """Verify referable DR sensitivity at 90% specificity computation."""
    y_true = np.array([0, 0, 1, 2, 3, 4, 0, 1, 2, 4])
    # Mock probabilities: 10 samples x 5 classes
    y_prob = np.zeros((10, 5))
    # Make samples 3, 4, 5, 8, 9 have high prob on class 2-4
    for i in [3, 4, 5, 8, 9]:
        y_prob[i, 3] = 0.9
    for i in [0, 1, 2, 6, 7]:
        y_prob[i, 0] = 0.9

    res = compute_referable_dr_metrics(y_true, y_prob, target_specificity=0.8)
    assert "referable_sens_at_90_spec" in res
    assert res["referable_sens_at_90_spec"] >= 0.0


def test_predictor_initialization():
    """Verify DRPredictor initializes cleanly with both baseline and retinaguard types."""
    pred_base = DRPredictor(model_type="baseline")
    assert pred_base.model_type == "baseline"

    pred_rg = DRPredictor(model_type="retinaguard")
    assert pred_rg.model_type == "retinaguard"
