"""RetinaGuard - Unit tests for preprocessing pipeline."""

import cv2
import numpy as np
import pytest

from src.data.preprocessing import RetinalPreprocessor
from src.data.create_mock_data import generate_synthetic_fundus


@pytest.fixture
def sample_fundus(tmp_path):
    """Create a temporary synthetic fundus image and mask for testing."""
    img, mask = generate_synthetic_fundus(height=400, width=400, dr_grade=2, seed=42)
    img_path = tmp_path / "test_fundus.jpg"
    mask_path = tmp_path / "test_mask.png"

    cv2.imwrite(str(img_path), img)
    cv2.imwrite(str(mask_path), mask)

    return img_path, mask_path


def test_preprocessor_default_config():
    """Verify preprocessor initializes with valid default config."""
    preprocessor = RetinalPreprocessor()
    assert preprocessor.config is not None
    assert preprocessor.config["fov_detection"]["enabled"] is True
    assert preprocessor.config["resize"]["target_size"] == [512, 512]


def test_process_image_end_to_end(sample_fundus, tmp_path):
    """Verify processing pipeline produces correct output dimensions and files."""
    img_path, mask_path = sample_fundus
    save_img = tmp_path / "out_img.jpg"
    save_mask = tmp_path / "out_mask.png"

    preprocessor = RetinalPreprocessor()
    meta = preprocessor.process_image(
        image_path=img_path,
        save_path=save_img,
        mask_path=mask_path,
        mask_save_path=save_mask,
    )

    assert meta["status"] == "processed"
    assert meta["processed_size"] == [512, 512]
    assert save_img.exists()
    assert save_mask.exists()

    # Read output and verify shapes
    out_img = cv2.imread(str(save_img))
    out_mask = cv2.imread(str(save_mask), cv2.IMREAD_GRAYSCALE)
    assert out_img.shape == (512, 512, 3)
    assert out_mask.shape == (512, 512)


def test_fov_crop(sample_fundus):
    """Verify FOV detection crops excess border margins."""
    img_path, _ = sample_fundus
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    preprocessor = RetinalPreprocessor()
    cropped, crop_info = preprocessor._detect_and_crop_fov(img)

    assert crop_info["x2"] > crop_info["x1"]
    assert crop_info["y2"] > crop_info["y1"]
    assert cropped.shape[0] <= h
    assert cropped.shape[1] <= w


def test_quality_assessment(sample_fundus):
    """Verify quality assessment runs without error and returns flags."""
    img_path, _ = sample_fundus
    img = cv2.imread(str(img_path))

    preprocessor = RetinalPreprocessor()
    quality = preprocessor._assess_quality(img)

    assert "flags" in quality
    assert isinstance(quality["flags"], list)
