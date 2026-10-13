"""RetinaGuard - Unit tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_dr_metrics,
    format_metrics_report,
    DR_GRADE_NAMES,
)


def test_perfect_predictions():
    """Verify metrics when predictions perfectly match ground truth."""
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    metrics = compute_dr_metrics(y_true, y_pred, num_classes=5)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["quadratic_weighted_kappa"] == 1.0
    assert metrics["num_samples"] == 10
    assert np.array(metrics["confusion_matrix"]).shape == (5, 5)


def test_metric_keys_present():
    """Verify all standard required metric keys are returned."""
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 2, 4])

    metrics = compute_dr_metrics(y_true, y_pred, num_classes=5)

    required_keys = [
        "accuracy", "macro_f1", "quadratic_weighted_kappa",
        "confusion_matrix", "per_class", "num_samples",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing metric key: {key}"

    for grade_idx in range(5):
        class_name = DR_GRADE_NAMES[grade_idx]
        assert class_name in metrics["per_class"]
        class_metrics = metrics["per_class"][class_name]
        for field in ["precision", "recall", "sensitivity", "specificity", "support"]:
            assert field in class_metrics, f"Missing field {field} in {class_name}"


def test_qwk_penalty():
    """Verify quadratic weighted kappa penalizes distant mistakes more than adjacent ones."""
    y_true = np.array([0, 0, 0, 0, 4, 4, 4, 4])

    # Adjacent error (0 predicted as 1)
    y_pred_close = np.array([1, 1, 1, 1, 3, 3, 3, 3])
    metrics_close = compute_dr_metrics(y_true, y_pred_close, num_classes=5)

    # Distant error (0 predicted as 4)
    y_pred_far = np.array([4, 4, 4, 4, 0, 0, 0, 0])
    metrics_far = compute_dr_metrics(y_true, y_pred_far, num_classes=5)

    assert metrics_close["quadratic_weighted_kappa"] > metrics_far["quadratic_weighted_kappa"]


def test_format_metrics_report():
    """Verify string report formatting."""
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    metrics = compute_dr_metrics(y_true, y_pred, num_classes=5)

    report_str = format_metrics_report(metrics, title="Test Report")
    assert "Test Report" in report_str
    assert "Accuracy" in report_str
    assert "QWK" in report_str
