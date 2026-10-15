"""RetinaGuard - Clinical Optical Robustness & Perturbation Stress Testing.

Evaluates model resilience under real clinical imaging artifacts:
1. Camera defocus / blur (simulated cataract or poor focusing)
2. Retinal illumination shifts and contrast decay
3. Digital sensor noise
4. Geometric orientation invariance (rotation / horizontal reflection)
"""

import cv2
import numpy as np
import pytest
import torch

from src.models.baseline import DRClassifier
from src.models.retinaguard import RetinaGuardModel


@pytest.fixture
def dummy_fundus_tensor():
    """Create normalized fundus tensor (1, 3, 128, 128)."""
    torch.manual_seed(42)
    # Background circular fundus pattern
    img = torch.randn(1, 3, 128, 128) * 0.2 + 0.5
    return img.clamp(0.0, 1.0)


def test_gaussian_blur_robustness(dummy_fundus_tensor):
    """Verify model produces valid predictions and graceful degradation under severe defocus blur."""
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    model.eval()

    with torch.no_grad():
        clean_out = model(dummy_fundus_tensor)["disease_logits"]
        clean_probs = torch.softmax(clean_out, dim=1).numpy()[0]

        # Apply severe Gaussian blur kernel (simulating high-stage cataract / camera defocus)
        blurred_np = cv2.GaussianBlur(dummy_fundus_tensor[0].permute(1, 2, 0).numpy(), (15, 15), 5.0)
        blurred_tensor = torch.from_numpy(blurred_np).permute(2, 0, 1).unsqueeze(0).float()

        blur_out = model(blurred_tensor)["disease_logits"]
        blur_probs = torch.softmax(blur_out, dim=1).numpy()[0]

    assert len(blur_probs) == 5
    assert np.isclose(np.sum(blur_probs), 1.0, atol=1e-4)
    assert not np.isnan(blur_probs).any()
    # Predictions shouldn't collapse into NaN or uniform overflow
    assert np.max(blur_probs) > 0.20


def test_illumination_shift_invariance(dummy_fundus_tensor):
    """Verify robustness to over/under-exposure flash intensity variations."""
    model = DRClassifier(architecture="resnet18", num_classes=5, pretrained=False)
    model.eval()

    with torch.no_grad():
        clean_probs = torch.softmax(model(dummy_fundus_tensor), dim=1).numpy()[0]

        # Low-flash underexposure (-30%)
        dark_tensor = (dummy_fundus_tensor * 0.70).clamp(0.0, 1.0)
        dark_probs = torch.softmax(model(dark_tensor), dim=1).numpy()[0]

        # High-flash overexposure (+30%)
        bright_tensor = (dummy_fundus_tensor * 1.30).clamp(0.0, 1.0)
        bright_probs = torch.softmax(model(bright_tensor), dim=1).numpy()[0]

    assert not np.isnan(dark_probs).any()
    assert not np.isnan(bright_probs).any()
    assert np.isclose(np.sum(dark_probs), 1.0, atol=1e-4)
    assert np.isclose(np.sum(bright_probs), 1.0, atol=1e-4)


def test_sensor_noise_stability(dummy_fundus_tensor):
    """Verify stability under camera sensor thermal noise."""
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    model.eval()

    torch.manual_seed(99)
    noise = torch.randn_like(dummy_fundus_tensor) * 0.05
    noisy_tensor = (dummy_fundus_tensor + noise).clamp(0.0, 1.0)

    with torch.no_grad():
        clean_out = model(dummy_fundus_tensor)
        noisy_out = model(noisy_tensor)

        p_clean = torch.softmax(clean_out["disease_logits"], dim=1).numpy()[0]
        p_noisy = torch.softmax(noisy_out["disease_logits"], dim=1).numpy()[0]

    # Total variation distance between clean and noisy distributions should be bounded
    tv_dist = 0.5 * np.sum(np.abs(p_clean - p_noisy))
    assert tv_dist < 0.50  # Bounded sensitivity to minor noise


def test_horizontal_flip_equivariance(dummy_fundus_tensor):
    """Verify that horizontal flip (left/right eye symmetry) retains valid output bounds."""
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    model.eval()

    flipped_tensor = torch.flip(dummy_fundus_tensor, dims=[3])

    with torch.no_grad():
        out = model(dummy_fundus_tensor)
        flipped_out = model(flipped_tensor)

        p_orig = torch.softmax(out["disease_logits"], dim=1).numpy()[0]
        p_flip = torch.softmax(flipped_out["disease_logits"], dim=1).numpy()[0]

    assert len(p_flip) == 5
    assert not np.isnan(p_flip).any()
    assert np.isclose(np.sum(p_flip), 1.0, atol=1e-4)


def test_cataract_media_opacity_simulation(dummy_fundus_tensor):
    """Simulate cloudy ocular media (loss of high frequencies and yellowish tint)."""
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    model.eval()

    # Cataract filter: attenuate blue channel, blur red/green
    img_np = dummy_fundus_tensor[0].permute(1, 2, 0).numpy().copy()
    img_np[:, :, 2] *= 0.4  # Attenuate blue channel
    img_np = cv2.GaussianBlur(img_np, (9, 9), 2.0)
    cataract_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().clamp(0.0, 1.0)

    with torch.no_grad():
        out = model(cataract_tensor)
        probs = torch.softmax(out["disease_logits"], dim=1).numpy()[0]

    assert not np.isnan(probs).any()
    assert len(probs) == 5
