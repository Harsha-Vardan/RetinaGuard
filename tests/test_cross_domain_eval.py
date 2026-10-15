"""RetinaGuard - Unit Tests for Cross-Domain Generalization Evaluator."""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.create_mock_data import generate_synthetic_fundus
from src.evaluation.evaluate_cross_domain import CrossDomainEvaluator
from src.models.baseline import DRClassifier
from src.models.retinaguard import RetinaGuardModel


@pytest.fixture
def dummy_test_df(tmp_path):
    """Create a minimal test DataFrame with synthetic images."""
    records = []
    for i in range(5):
        img, _ = generate_synthetic_fundus(height=128, width=128, dr_grade=i, seed=42 + i)
        img_path = tmp_path / f"test_{i}.png"
        cv2.imwrite(str(img_path), img)
        records.append({
            "image_id": f"img_{i}",
            "image_path": str(img_path),
            "dr_grade": i,
            "dataset_id": "APTOS",
        })
    return pd.DataFrame(records)


def test_cross_domain_evaluator_init():
    """Verify evaluator initializes output directories."""
    evaluator = CrossDomainEvaluator()
    assert evaluator.results_dir.exists()
    assert evaluator.figures_dir.exists()


def test_cross_domain_evaluator_empty_dataset():
    """Verify graceful handling when dataset DataFrame is empty."""
    evaluator = CrossDomainEvaluator()
    empty_df = pd.DataFrame({"image_path": [], "dr_grade": []})
    model = DRClassifier(architecture="resnet18", num_classes=5, pretrained=False)

    res = evaluator.evaluate_model(model, empty_df, dataset_name="empty_cohort")
    assert "error" in res
    assert res["error"] == "no_valid_samples"


def test_cross_domain_evaluator_evaluate_baseline(dummy_test_df):
    """Verify evaluation loop computes metrics and returns results dict for Baseline."""
    evaluator = CrossDomainEvaluator()
    model = DRClassifier(architecture="resnet18", num_classes=5, pretrained=False)

    results = evaluator.evaluate_model(
        model=model,
        test_df=dummy_test_df,
        dataset_name="test_cohort",
        image_size=128,
        batch_size=2,
    )

    assert "dataset" in results
    assert results["dataset"] == "test_cohort"
    assert "accuracy" in results
    assert "quadratic_weighted_kappa" in results
    assert "macro_f1" in results
    assert results["num_samples"] == len(dummy_test_df)
    assert "bootstrap_ci_95" in results


def test_cross_domain_evaluator_evaluate_retinaguard(dummy_test_df):
    """Verify evaluation loop unpacks dictionary logits and computes metrics for RetinaGuardModel."""
    evaluator = CrossDomainEvaluator()
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)

    results = evaluator.evaluate_model(
        model=model,
        test_df=dummy_test_df,
        dataset_name="retinaguard_test",
        image_size=128,
        batch_size=2,
        model_name="RetinaGuard",
    )

    assert "dataset" in results
    assert results["dataset"] == "retinaguard_test"
    assert "accuracy" in results
    assert "quadratic_weighted_kappa" in results
    assert "bootstrap_ci_95" in results
    assert "quadratic_weighted_kappa" in results["bootstrap_ci_95"]
