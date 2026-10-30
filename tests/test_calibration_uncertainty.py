"""RetinaGuard - Tests for Model Calibration and Clinical Uncertainty Quantification."""

import numpy as np
import pytest

from src.evaluation.calibration import (
    compute_expected_calibration_error,
    compute_brier_score,
    compute_uncertainty_metrics,
    ConformalPredictor,
)


def test_ece_perfect_calibration():
    """Verify that a perfectly calibrated model yields ECE == 0.0."""
    # 100 samples with confidence 1.0 that are 100% accurate
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.zeros((100, 2))
    y_prob[:50, 0] = 1.0
    y_prob[50:, 1] = 1.0

    res = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
    assert res["ece"] == 0.0
    assert res["max_calibration_error"] == 0.0


def test_ece_severe_overconfidence():
    """Verify that an overconfident model has non-zero ECE matching error rate."""
    # Predict 1.0 confidence for class 0, but true labels are all class 1 (0% accuracy)
    y_true = np.ones(50, dtype=int)
    y_prob = np.zeros((50, 2))
    y_prob[:, 0] = 1.0  # Confident 1.0 in wrong class

    res = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert res["ece"] == 1.0
    assert res["max_calibration_error"] == 1.0


def test_ece_bin_structure():
    """Verify bin outputs and structure."""
    np.random.seed(42)
    y_true = np.random.randint(0, 5, size=100)
    y_logits = np.random.randn(100, 5)
    y_prob = np.exp(y_logits) / np.sum(np.exp(y_logits), axis=1, keepdims=True)

    res = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
    assert len(res["bin_accuracies"]) == 10
    assert len(res["bin_confidences"]) == 10
    assert len(res["bin_counts"]) == 10
    assert sum(res["bin_counts"]) == 100
    assert 0.0 <= res["ece"] <= 1.0


def test_ece_empty_input():
    """Verify graceful handling of empty inputs."""
    res = compute_expected_calibration_error(np.array([]), np.empty((0, 5)))
    assert res["ece"] == 0.0
    assert res["bin_counts"] == []


def test_brier_score_perfect():
    """Verify Brier score is 0.0 for deterministic correct predictions."""
    y_true = np.array([0, 1, 2, 3, 4])
    y_prob = np.eye(5)
    bs = compute_brier_score(y_true, y_prob, num_classes=5)
    assert bs == 0.0


def test_brier_score_worst():
    """Verify Brier score is 2.0 when predicting 1.0 for wrong class."""
    y_true = np.array([0])
    y_prob = np.array([[0.0, 1.0]])  # Truth is 0, predicted 1
    bs = compute_brier_score(y_true, y_prob, num_classes=2)
    assert bs == 2.0


def test_brier_score_uniform():
    """Verify Brier score for uniform random guessing on 5 classes."""
    y_true = np.array([0, 1, 2, 3, 4])
    y_prob = np.ones((5, 5)) * 0.2
    # For each sample: (0.8)^2 + 4 * (0.2)^2 = 0.64 + 0.16 = 0.8
    bs = compute_brier_score(y_true, y_prob, num_classes=5)
    assert np.isclose(bs, 0.8, atol=1e-3)


def test_uncertainty_metrics():
    """Verify normalized entropy and low-confidence fraction."""
    # Deterministic predictions -> Entropy must be 0.0
    y_prob_det = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    res_det = compute_uncertainty_metrics(y_prob_det, confidence_threshold=0.7)
    assert res_det["mean_entropy"] == 0.0
    assert res_det["mean_confidence"] == 1.0
    assert res_det["low_confidence_fraction"] == 0.0

    # Uniform predictions -> Entropy must be 1.0, low confidence fraction 1.0
    y_prob_unif = np.ones((10, 5)) * 0.2
    res_unif = compute_uncertainty_metrics(y_prob_unif, confidence_threshold=0.5)
    assert np.isclose(res_unif["mean_entropy"], 1.0, atol=1e-3)
    assert np.isclose(res_unif["mean_confidence"], 0.2, atol=1e-3)
    assert res_unif["low_confidence_fraction"] == 1.0


def test_uncertainty_metrics_empty():
    """Verify uncertainty calculation on empty array."""
    res = compute_uncertainty_metrics(np.empty((0, 5)))
    assert res["mean_entropy"] == 0.0
    assert res["mean_confidence"] == 0.0


def test_conformal_prediction_coverage_guarantee():
    """Verify that ConformalPredictor satisfies the finite-sample coverage bound P(Y in C(X)) >= 1 - alpha."""
    np.random.seed(42)
    n_cal = 200
    n_test = 200
    n_classes = 5

    # Simulated validation and test probability distributions
    val_labels = np.random.randint(0, n_classes, size=n_cal)
    val_probs = np.random.dirichlet(np.ones(n_classes), size=n_cal)
    # Give some predictive signal to true class
    for i in range(n_cal):
        val_probs[i, val_labels[i]] += 1.5
    val_probs /= val_probs.sum(axis=1, keepdims=True)

    test_labels = np.random.randint(0, n_classes, size=n_test)
    test_probs = np.random.dirichlet(np.ones(n_classes), size=n_test)
    for i in range(n_test):
        test_probs[i, test_labels[i]] += 1.5
    test_probs /= test_probs.sum(axis=1, keepdims=True)

    alpha = 0.10
    cp = ConformalPredictor(alpha=alpha)
    q_hat = cp.calibrate(val_probs, val_labels)

    assert 0.0 <= q_hat <= 1.0
    metrics = cp.evaluate_coverage(test_probs, test_labels)

    assert metrics["coverage_guarantee_met"] is True
    assert metrics["empirical_coverage"] >= 0.85
    assert 1.0 <= metrics["mean_set_size"] <= n_classes
    assert 0.0 <= metrics["singleton_rate"] <= 1.0


def test_conformal_prediction_single_sample():
    """Verify prediction set generation for a single probability vector."""
    cp = ConformalPredictor(alpha=0.10)
    cp.q_hat = 0.30
    cp.calibrated = True

    # Threshold = 1 - 0.30 = 0.70
    prob_clear = np.array([0.85, 0.05, 0.05, 0.03, 0.02])
    c_set_clear = cp.predict_set(prob_clear)
    assert c_set_clear == [0]

    # Borderline sample: Grade 0 (40%) vs Grade 1 (40%)
    prob_borderline = np.array([0.40, 0.40, 0.10, 0.05, 0.05])
    cp.q_hat = 0.75  # threshold = 0.25
    c_set_border = cp.predict_set(prob_borderline)
    assert 0 in c_set_border and 1 in c_set_border

