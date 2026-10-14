"""RetinaGuard - Tests for Data Ingestion, Auditing, and Pipeline Integrity."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.audit_dataset import DatasetAuditor
from src.data.create_mock_data import generate_synthetic_fundus
from src.data.preprocessing import RetinalPreprocessor


@pytest.fixture
def auditor():
    return DatasetAuditor()


@pytest.fixture
def preprocessor():
    return RetinalPreprocessor()


def test_dataset_auditor_initialization(auditor):
    """Verify DatasetAuditor initializes valid DR grade standards."""
    assert auditor.VALID_DR_GRADES == {0, 1, 2, 3, 4}
    assert ".jpg" in auditor.IMAGE_EXTENSIONS
    assert ".png" in auditor.IMAGE_EXTENSIONS


def test_preprocessor_assess_quality_normal(preprocessor):
    """Verify quality check on standard synthetic retinal image."""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 128
    res = preprocessor._assess_quality(img)
    assert isinstance(res, dict)
    assert "flags" in res
    assert "too_small" not in res["flags"]


def test_preprocessor_assess_quality_too_small(preprocessor):
    """Verify image below minimum dimension is flagged."""
    tiny_img = np.ones((50, 50, 3), dtype=np.uint8) * 100
    res = preprocessor._assess_quality(tiny_img)
    assert "too_small" in res["flags"]
    assert res.get("min_dim_fail") is True


def test_preprocessor_assess_quality_dark_and_bright(preprocessor):
    """Verify severely underexposed and overexposed images are flagged."""
    dark_img = np.ones((200, 200, 3), dtype=np.uint8) * 10
    res_dark = preprocessor._assess_quality(dark_img)
    assert "too_dark" in res_dark["flags"]

    bright_img = np.ones((200, 200, 3), dtype=np.uint8) * 250
    res_bright = preprocessor._assess_quality(bright_img)
    assert "too_bright" in res_bright["flags"]


def test_generate_synthetic_fundus_all_grades():
    """Verify synthetic fundus generator creates valid images and masks across grades 0 to 4."""
    for grade in range(5):
        img, mask = generate_synthetic_fundus(height=256, width=256, dr_grade=grade, seed=42 + grade)
        assert img.shape == (256, 256, 3)
        assert img.dtype == np.uint8
        assert mask.shape == (256, 256)
        assert mask.dtype == np.uint8
        assert img.max() > 0

        # Unique values in mask should be subset of {0, 255}
        unique_vals = set(np.unique(mask))
        assert unique_vals.issubset({0, 255})
