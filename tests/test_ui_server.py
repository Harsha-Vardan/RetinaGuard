"""RetinaGuard - Tests for UI Server, Model Engine, and REST Endpoints."""

import base64
import json
from http import HTTPStatus
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from src.ui.server import DR_GRADE_METADATA, ModelEngine, RetinaGuardRequestHandler


@pytest.fixture
def dummy_fundus_image():
    """Generates an in-memory synthetic 512x512 fundus image encoded as bytes."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 200, (20, 40, 180), -1)  # Orange-red retinal disk
    cv2.circle(img, (200, 256), 30, (50, 200, 220), -1)  # Optic disc
    cv2.circle(img, (300, 256), 10, (10, 10, 80), -1)    # Macula
    _, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


@pytest.fixture
def engine():
    """Initializes ModelEngine for testing."""
    return ModelEngine()


def test_dr_grade_metadata_integrity():
    """Verify that all 5 ICDR grades are fully specified with descriptions."""
    assert len(DR_GRADE_METADATA) == 5
    for grade in range(5):
        assert grade in DR_GRADE_METADATA
        meta = DR_GRADE_METADATA[grade]
        assert "title" in meta
        assert "icdr" in meta
        assert "severity" in meta
        assert "referable" in meta
        assert "recommendation" in meta

    assert DR_GRADE_METADATA[0]["referable"] is False
    assert DR_GRADE_METADATA[1]["referable"] is False
    assert DR_GRADE_METADATA[2]["referable"] is True
    assert DR_GRADE_METADATA[3]["referable"] is True
    assert DR_GRADE_METADATA[4]["referable"] is True


def test_model_engine_initialization(engine):
    """Verify that ModelEngine correctly initializes hardware and preprocessor."""
    assert engine.device is not None
    assert engine.preprocessor is not None
    assert engine.transform is not None


def test_model_engine_get_sample_images(engine):
    """Verify that sample images list is returned with required keys."""
    samples = engine.get_sample_images()
    assert isinstance(samples, list)
    if len(samples) > 0:
        s = samples[0]
        assert "id" in s
        assert "filename" in s
        assert "dataset" in s
        assert "description" in s
        assert "rel_path" in s


def test_to_base64_png(engine):
    """Verify conversion of RGB numpy array to base64 data URI."""
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    b64_uri = engine._to_base64_png(rgb)
    assert b64_uri.startswith("data:image/png;base64,")
    # Decode to verify round-trip
    b64_data = b64_uri.split(",")[1]
    decoded = base64.b64decode(b64_data)
    assert len(decoded) > 0


def test_process_and_predict_retinaguard(engine, dummy_fundus_image):
    """Verify prediction pipeline using RetinaGuard model."""
    res = engine.process_and_predict(dummy_fundus_image, model_type="retinaguard")
    assert "model_type" in res
    assert res["model_key"] == "retinaguard"
    assert "predicted_grade" in res
    assert 0 <= res["predicted_grade"] <= 4
    assert "title" in res
    assert "icdr_description" in res
    assert "referable_dr" in res
    assert "confidence" in res
    assert "probabilities" in res
    assert len(res["probabilities"]) == 5
    assert "images" in res
    assert "raw" in res["images"]
    assert "preprocessed" in res["images"]
    assert "heatmap" in res["images"]
    assert "overlay" in res["images"]


def test_process_and_predict_baseline(engine, dummy_fundus_image):
    """Verify prediction pipeline using Baseline model."""
    res = engine.process_and_predict(dummy_fundus_image, model_type="baseline")
    assert "model_type" in res
    assert res["model_key"] == "baseline"
    assert "predicted_grade" in res
    assert 0 <= res["predicted_grade"] <= 4
    assert res["images"]["heatmap"].startswith("data:image/png;base64,")


def test_process_and_predict_invalid_image(engine):
    """Verify ValueError is raised when corrupted image bytes are passed."""
    with pytest.raises(ValueError, match="Invalid image file format"):
        engine.process_and_predict(b"corrupted_non_image_data")
