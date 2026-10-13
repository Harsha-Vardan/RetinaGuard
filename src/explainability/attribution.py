"""RetinaGuard - Saliency and Clinical Attribution Module.

Provides:
1. Grad-CAM visual explanations for arbitrary CNN backbones.
2. Direct Lesion Attention map extraction from RetinaGuardModel.
3. Attribution IoU and Dice metric evaluation against ground-truth clinical lesion masks.
4. Publication-ready multi-panel visualization generator.

Usage:
    from src.explainability.attribution import (
        GradCAM,
        extract_lesion_attention_map,
        compute_attribution_iou,
        generate_explanation_figure,
    )
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for PyTorch models."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """Initialize Grad-CAM hook.

        Args:
            model: PyTorch classification model.
            target_layer: Target convolutional layer for feature maps (e.g. model.backbone.layer4).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Compute Grad-CAM heatmap for a single input tensor.

        Args:
            input_tensor: Tensor of shape (1, 3, H, W).
            target_class: Target class index. If None, highest scoring class is chosen.

        Returns:
            2D numpy array heatmap in [0.0, 1.0] of shape (H, W).
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if isinstance(output, dict):
            logits = output.get("disease_logits", output.get("logits"))
        else:
            logits = output

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        # Global average pooling on gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Upsample to input resolution
        _, _, h, w = input_tensor.shape
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


def extract_lesion_attention_map(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Extract spatial lesion attention map directly from RetinaGuardModel.

    Args:
        model: RetinaGuardModel instance.
        input_tensor: Input tensor of shape (1, 3, H, W).
        target_size: Optional (height, width) to upsample attention map.

    Returns:
        2D numpy array attention map in [0.0, 1.0].
    """
    model.eval()
    with torch.no_grad():
        out = model(input_tensor)
        if isinstance(out, dict) and "attn_mask" in out:
            attn = out["attn_mask"]  # (1, 1, H', W')
        elif hasattr(model, "lesion_attention"):
            feat = model.extract_features(input_tensor)
            _, attn = model.lesion_attention(feat)
        else:
            raise ValueError("Model does not provide an attention mask.")

        if target_size is not None:
            attn = F.interpolate(attn, size=target_size, mode="bilinear", align_corners=False)
        else:
            _, _, h, w = input_tensor.shape
            attn = F.interpolate(attn, size=(h, w), mode="bilinear", align_corners=False)

        heatmap = attn.squeeze().cpu().numpy().astype(np.float32)
        h_min, h_max = np.min(heatmap), np.max(heatmap)
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min + 1e-8)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap


def compute_attribution_iou(
    attribution_map: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute Intersection-over-Union (IoU) between attribution map and clinical mask.

    Args:
        attribution_map: 2D numpy array in [0.0, 1.0].
        ground_truth_mask: 2D numpy array binary mask or [0, 255].
        threshold: Binarization threshold for attribution map.

    Returns:
        IoU score in [0.0, 1.0].
    """
    pred_bin = (attribution_map >= threshold).astype(bool)
    gt_bin = (ground_truth_mask > 0).astype(bool)

    if pred_bin.shape != gt_bin.shape:
        # Resize gt_bin to match attribution map shape
        gt_bin = cv2.resize(
            gt_bin.astype(np.uint8),
            (pred_bin.shape[1], pred_bin.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        # Both masks are completely empty: perfect agreement
        return 1.0

    return float(intersection / (union + 1e-8))


def compute_attribution_dice(
    attribution_map: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute Dice similarity coefficient between attribution map and clinical mask.

    Args:
        attribution_map: 2D float array in [0.0, 1.0].
        ground_truth_mask: 2D array binary mask.
        threshold: Threshold for binarization.

    Returns:
        Dice coefficient in [0.0, 1.0].
    """
    pred_bin = (attribution_map >= threshold).astype(bool)
    gt_bin = (ground_truth_mask > 0).astype(bool)

    if pred_bin.shape != gt_bin.shape:
        gt_bin = cv2.resize(
            gt_bin.astype(np.uint8),
            (pred_bin.shape[1], pred_bin.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    total = pred_bin.sum() + gt_bin.sum()

    if total == 0:
        return 1.0

    return float((2.0 * intersection) / (total + 1e-8))


def generate_explanation_figure(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    ground_truth_mask: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    predicted_grade: Optional[int] = None,
    confidence: Optional[float] = None,
    iou: Optional[float] = None,
) -> plt.Figure:
    """Generate a multi-panel visual explanation figure.

    Panel 1: Original fundus photograph.
    Panel 2: Heatmap (Grad-CAM or Lesion Attention).
    Panel 3: Heatmap overlay on fundus image.
    Panel 4: Clinical ground truth mask / alignment overlay (if available).

    Args:
        image_np: Original RGB image array (H, W, 3) in [0, 255] or [0, 1].
        heatmap: 2D float array in [0.0, 1.0].
        ground_truth_mask: Optional binary mask (H, W).
        save_path: Path to save figure to.
        predicted_grade: Predicted DR grade (0-4).
        confidence: Prediction confidence score.
        iou: Optional computed Attribution IoU.

    Returns:
        matplotlib Figure object.
    """
    if image_np.max() <= 1.0:
        image_rgb = (image_np * 255).astype(np.uint8)
    else:
        image_rgb = image_np.astype(np.uint8)

    h, w = image_rgb.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Colorize heatmap with JET colormap
    heatmap_u8 = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    color_cam = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    color_cam = cv2.cvtColor(color_cam, cv2.COLOR_BGR2RGB)

    # Blended overlay
    overlay = cv2.addWeighted(image_rgb, 0.6, color_cam, 0.4, 0)

    num_panels = 4 if ground_truth_mask is not None else 3
    fig, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 4.5), dpi=150)

    # Panel 1: Original Fundus
    axes[0].imshow(image_rgb)
    title_0 = "Original Fundus"
    if predicted_grade is not None:
        title_0 += f"\nPred: Grade {predicted_grade}"
        if confidence is not None:
            title_0 += f" ({confidence*100:.1f}%)"
    axes[0].set_title(title_0, fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Attention / Saliency Map
    im2 = axes[1].imshow(heatmap, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Attribution Heatmap", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Blended Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Pathology Overlay", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # Panel 4: Clinical Ground Truth or Alignment
    if ground_truth_mask is not None:
        if ground_truth_mask.shape != (h, w):
            gt_disp = cv2.resize(
                ground_truth_mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            gt_disp = ground_truth_mask

        # Show GT mask as green overlay on grayscale fundus
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        gt_overlay = gray_rgb.copy()
        gt_overlay[gt_disp > 0] = [0, 255, 0]  # Green for clinical lesions
        blended_gt = cv2.addWeighted(gray_rgb, 0.5, gt_overlay, 0.5, 0)

        axes[3].imshow(blended_gt)
        title_3 = "Clinical Lesion GT (Green)"
        if iou is not None:
            title_3 += f"\nAttribution IoU: {iou:.3f}"
        axes[3].set_title(title_3, fontsize=11, fontweight="bold")
        axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return fig
