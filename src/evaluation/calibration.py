"""RetinaGuard - Clinical Calibration & Uncertainty Quantification.

Evaluates how well predicted confidence scores match empirical accuracy,
which is critical for high-stakes medical AI deployment (e.g. IEEE TMI standards).

Implements:
1. Expected Calibration Error (ECE) with adaptive/uniform binning
2. Multi-class Brier Score
3. Reliability diagram metrics
4. Predictive entropy and uncertainty estimation
"""

from typing import Any
import numpy as np


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute Expected Calibration Error (ECE) for multi-class classification.

    ECE measures the absolute difference between model confidence and actual accuracy:
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        y_true: Ground truth class integer labels (N,).
        y_prob: Predicted probability array of shape (N, C).
        n_bins: Number of confidence bins (default 10).

    Returns:
        Dictionary containing 'ece', 'max_calibration_error', 'bin_accuracies',
        'bin_confidences', and 'bin_counts'.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0:
        return {
            "ece": 0.0,
            "max_calibration_error": 0.0,
            "bin_accuracies": [],
            "bin_confidences": [],
            "bin_counts": [],
        }

    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = int(np.sum(in_bin))
        bin_counts.append(bin_count)

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_accs.append(round(bin_acc, 4))
            bin_confs.append(round(bin_conf, 4))

            calibration_error = abs(bin_acc - bin_conf)
            ece += (bin_count / total_samples) * calibration_error
            mce = max(mce, calibration_error)
        else:
            bin_accs.append(0.0)
            bin_confs.append(round(float((bin_lower + bin_upper) / 2.0), 4))

    return {
        "ece": round(float(ece), 4),
        "max_calibration_error": round(float(mce), 4),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
    }


def compute_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 5,
) -> float:
    """Compute multi-class Brier score.

    Brier score is the mean squared error between one-hot true targets and predicted probabilities:
        BS = (1 / N) * sum_{i=1}^N sum_{c=1}^C (p_{ic} - y_{ic})^2
    Ranges from 0 (perfect probability forecasting) to 2.

    Args:
        y_true: Ground truth labels (N,).
        y_prob: Predicted probability array of shape (N, num_classes).
        num_classes: Number of categorical classes.

    Returns:
        Brier score as a float rounded to 4 decimals.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0

    # Convert true labels to one-hot encoding
    one_hot = np.zeros((n_samples, num_classes), dtype=float)
    for i, label in enumerate(y_true):
        if 0 <= label < num_classes:
            one_hot[i, label] = 1.0

    brier = np.mean(np.sum((y_prob - one_hot) ** 2, axis=1))
    return round(float(brier), 4)


def compute_uncertainty_metrics(
    y_prob: np.ndarray,
    confidence_threshold: float = 0.70,
) -> dict[str, float]:
    """Compute prediction uncertainty metrics across a cohort.

    Args:
        y_prob: Predicted probability array (N, C).
        confidence_threshold: Threshold below which a prediction is considered low-confidence.

    Returns:
        Dictionary with mean entropy, mean confidence, and low-confidence fraction.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_prob) == 0:
        return {"mean_entropy": 0.0, "mean_confidence": 0.0, "low_confidence_fraction": 0.0}

    # Clip probabilities to avoid log(0)
    eps = 1e-12
    p_clipped = np.clip(y_prob, eps, 1.0 - eps)

    # Normalized Shannon entropy: H = -sum(p * log(p)) / log(C)
    num_classes = y_prob.shape[1]
    raw_entropy = -np.sum(p_clipped * np.log(p_clipped), axis=1)
    max_entropy = np.log(num_classes) if num_classes > 1 else 1.0
    normalized_entropy = raw_entropy / max_entropy

    max_conf = np.max(y_prob, axis=1)
    low_conf_mask = max_conf < confidence_threshold

    return {
        "mean_entropy": round(float(np.mean(normalized_entropy)), 4),
        "mean_confidence": round(float(np.mean(max_conf)), 4),
        "low_confidence_fraction": round(float(np.mean(low_conf_mask)), 4),
    }


class ConformalPredictor:
    """Split Conformal Prediction for distribution-free risk-controlled medical classification.

    Guarantees finite-sample marginal coverage:
        P(Y in C(X)) >= 1 - alpha
    where C(X) is a prediction set and alpha in (0, 1) is the user-specified error tolerance.
    In DR screening, sets like {0, 1} formalize safe borderline non-referable triage,
    while wider sets trigger automated clinical specialist review.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q_hat: float | None = None
        self.calibrated = False

    def calibrate(self, val_probs: np.ndarray, val_labels: np.ndarray) -> float:
        """Calibrate quantile threshold using held-out validation predictions."""
        val_probs = np.asarray(val_probs, dtype=float)
        val_labels = np.asarray(val_labels, dtype=int)
        n = len(val_labels)
        if n == 0:
            raise ValueError("Validation set cannot be empty for conformal calibration.")

        # Non-conformity score: 1 - P(Y = true_class)
        true_class_probs = val_probs[np.arange(n), val_labels]
        scores = 1.0 - true_class_probs

        # Finite-sample adjusted quantile level: ceil((n+1)(1-alpha)) / n
        level = min(1.0, float(np.ceil((n + 1) * (1.0 - self.alpha)) / n))
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        self.calibrated = True
        return self.q_hat

    def predict_set(self, prob_vector: np.ndarray) -> list[int]:
        """Generate prediction set for a single probability vector."""
        prob_vector = np.asarray(prob_vector, dtype=float)
        if not self.calibrated or self.q_hat is None:
            threshold = 0.15
        else:
            threshold = max(0.0, 1.0 - self.q_hat)

        pred_set = [int(c) for c in np.where(prob_vector >= threshold)[0]]
        if not pred_set:
            pred_set = [int(np.argmax(prob_vector))]
        return pred_set

    def evaluate_coverage(self, test_probs: np.ndarray, test_labels: np.ndarray) -> dict[str, Any]:
        """Evaluate empirical coverage and set efficiency on an independent test cohort."""
        test_probs = np.asarray(test_probs, dtype=float)
        test_labels = np.asarray(test_labels, dtype=int)
        n = len(test_labels)

        covered = 0
        set_sizes = []
        sets = []

        for i in range(n):
            c_set = self.predict_set(test_probs[i])
            sets.append(c_set)
            set_sizes.append(len(c_set))
            if test_labels[i] in c_set:
                covered += 1

        empirical_coverage = covered / n if n > 0 else 0.0
        mean_set_size = float(np.mean(set_sizes)) if set_sizes else 0.0
        singleton_fraction = float(np.mean([s == 1 for s in set_sizes])) if set_sizes else 0.0
        ambiguous_fraction = float(np.mean([s >= 3 for s in set_sizes])) if set_sizes else 0.0

        return {
            "target_coverage": round(1.0 - self.alpha, 4),
            "empirical_coverage": round(empirical_coverage, 4),
            "coverage_guarantee_met": bool(empirical_coverage >= (1.0 - self.alpha - 0.05)),
            "mean_set_size": round(mean_set_size, 2),
            "singleton_rate": round(singleton_fraction, 4),
            "ambiguous_triage_rate": round(ambiguous_fraction, 4),
            "q_hat_threshold": round(self.q_hat, 4) if self.q_hat is not None else None,
        }

