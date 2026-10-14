"""RetinaGuard - Comprehensive Unit Tests for Inference Pipeline & DRPredictor."""

from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from src.data.create_mock_data import generate_synthetic_fundus
from src.inference.predict import DRPredictor, DR_GRADE_DESCRIPTIONS
from src.models.retinaguard import RetinaGuardModel
from src.models.baseline import DRClassifier


@pytest.fixture
def sample_fundus_image(tmp_path):
    """Create a temporary synthetic fundus image and return path."""
    img, mask = generate_synthetic_fundus(height=256, width=256, dr_grade=2, seed=101)
    img_path = tmp_path / "test_fundus.png"
    mask_path = tmp_path / "test_mask.png"
    cv2.imwrite(str(img_path), img)
    cv2.imwrite(str(mask_path), mask)
    return img_path, mask_path


def test_dr_grade_descriptions_completeness():
    """Verify all 5 ICDR DR grades (0-4) have clinical descriptions."""
    for grade in range(5):
        assert grade in DR_GRADE_DESCRIPTIONS
        assert len(DR_GRADE_DESCRIPTIONS[grade]) > 0


def test_dr_predictor_init_baseline():
    """Verify DRPredictor initializes with baseline DRClassifier."""
    predictor = DRPredictor(model_type="baseline")
    assert predictor.model_type == "baseline"
    assert isinstance(predictor.model, DRClassifier)


def test_dr_predictor_init_retinaguard():
    """Verify DRPredictor initializes with unified RetinaGuardModel."""
    predictor = DRPredictor(model_type="retinaguard")
    assert predictor.model_type == "retinaguard"
    assert isinstance(predictor.model, RetinaGuardModel)


def test_dr_predictor_predict_output_structure(sample_fundus_image):
    """Verify prediction result dictionary contains all clinical keys."""
    img_path, _ = sample_fundus_image
    predictor = DRPredictor(model_type="retinaguard")

    result = predictor.predict(img_path)

    assert "predicted_grade" in result
    assert "grade_description" in result
    assert "confidence" in result
    assert "referable_dr" in result
    assert "probabilities" in result

    assert result["predicted_grade"] in {0, 1, 2, 3, 4}
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["referable_dr"], bool)

    # Probabilities should have 5 grades and sum to approx 1.0
    probs = result["probabilities"]
    assert len(probs) == 5
    prob_sum = sum(probs.values())
    assert prob_sum == pytest.approx(1.0, abs=1e-3)


def test_dr_predictor_referable_consistency(sample_fundus_image):
    """Verify referable_dr flag matches grade >= 2 definition."""
    img_path, _ = sample_fundus_image
    predictor = DRPredictor(model_type="baseline")
    result = predictor.predict(img_path)

    expected_referable = bool(result["predicted_grade"] >= 2)
    assert result["referable_dr"] == expected_referable


def test_dr_predictor_nonexistent_image():
    """Verify appropriate error return when image does not exist."""
    predictor = DRPredictor(model_type="baseline")
    result = predictor.predict("nonexistent_image_12345.png")

    assert "error" in result
    assert "not found" in result["error"].lower()


def test_dr_predictor_corrupt_image(tmp_path):
    """Verify graceful error handling for corrupt/unreadable files."""
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_text("not an image")

    predictor = DRPredictor(model_type="baseline")
    result = predictor.predict(corrupt_file)

    assert "error" in result
    assert "cannot read image" in result["error"].lower()


def test_dr_predictor_with_heatmap_generation(sample_fundus_image, tmp_path):
    """Verify explanation figure generation and saving."""
    img_path, mask_path = sample_fundus_image
    save_fig_path = tmp_path / "explanation.png"

    predictor = DRPredictor(model_type="retinaguard")
    result = predictor.predict(
        img_path,
        save_heatmap_path=save_fig_path,
        ground_truth_mask_path=mask_path,
    )

    assert "explanation_figure" in result
    assert save_fig_path.exists()
    assert save_fig_path.stat().st_size > 0


def test_dr_predictor_auto_detect_baseline_checkpoint(tmp_path):
    """Verify auto-detection identifies baseline weights."""
    model = DRClassifier(architecture="resnet18", num_classes=5, pretrained=False)
    ckpt_path = tmp_path / "baseline_ckpt.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    predictor = DRPredictor(checkpoint_path=ckpt_path, model_type="auto")
    assert predictor.model_type == "baseline"


def test_dr_predictor_auto_detect_retinaguard_checkpoint(tmp_path):
    """Verify auto-detection identifies RetinaGuard weights by module keys."""
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    ckpt_path = tmp_path / "rg_ckpt.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    predictor = DRPredictor(checkpoint_path=ckpt_path, model_type="auto")
    assert predictor.model_type == "retinaguard"
