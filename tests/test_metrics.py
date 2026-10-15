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


def test_compute_referable_dr_metrics():
    """Verify referable DR sensitivity at target specificity calculation."""
    from src.evaluation.metrics import compute_referable_dr_metrics

    # 10 samples: 5 non-referable (grades 0, 1), 5 referable (grades 2, 3, 4)
    y_true = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 4])

    # Well-separated probabilities
    y_prob = np.zeros((10, 5))
    for i in range(5):
        y_prob[i, 0] = 0.8
        y_prob[i, 1] = 0.2
    for i in range(5, 10):
        y_prob[i, 2] = 0.5
        y_prob[i, 3] = 0.3
        y_prob[i, 4] = 0.2

    ref_metrics = compute_referable_dr_metrics(y_true, y_prob, target_specificity=0.90)
    assert "referable_sens_at_90_spec" in ref_metrics
    assert "referable_achieved_spec" in ref_metrics
    assert ref_metrics["referable_sens_at_90_spec"] == 1.0
    assert ref_metrics["referable_achieved_spec"] >= 0.90


def test_save_metrics_io(tmp_path):
    """Verify save_metrics writes valid JSON file that can be parsed."""
    import json
    from src.evaluation.metrics import save_metrics

    dummy_metrics = {"accuracy": 0.88, "qwk": 0.85, "samples": 100}
    json_path = tmp_path / "subdir" / "metrics.json"

    save_metrics(dummy_metrics, json_path)
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["accuracy"] == 0.88
    assert loaded["qwk"] == 0.85


def test_compute_bootstrap_ci():
    """Verify non-parametric bootstrap confidence interval estimation."""
    from src.evaluation.metrics import compute_bootstrap_ci

    y_true = np.array([0, 1, 2, 3, 4] * 10)
    y_pred = np.array([0, 1, 2, 3, 4] * 10)

    # 50 iterations for fast unit test
    ci_results = compute_bootstrap_ci(y_true, y_pred, n_bootstraps=50, confidence_level=0.95)

    assert "quadratic_weighted_kappa" in ci_results
    assert "accuracy" in ci_results
    assert "macro_f1" in ci_results
    assert ci_results["accuracy"]["mean"] == pytest.approx(1.0)
    assert ci_results["accuracy"]["ci_lower"] == pytest.approx(1.0)
    assert ci_results["accuracy"]["ci_upper"] == pytest.approx(1.0)
    assert ci_results["quadratic_weighted_kappa"]["mean"] == pytest.approx(1.0)


