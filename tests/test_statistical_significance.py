"""RetinaGuard - Statistical Significance and Hypothesis Testing.

Standardizes inferential statistical tests expected in medical AI publications
(e.g., Wilcoxon signed-rank test, paired permutation tests, and bootstrap coverage).
"""

import numpy as np
import pytest
from scipy import stats

from src.evaluation.metrics import compute_bootstrap_ci


def test_wilcoxon_signed_rank_paired_models():
    """Verify paired Wilcoxon signed-rank test for significance between model predictions."""
    np.random.seed(42)
    # 50 patients: RetinaGuard consistently has equal or lower absolute grading error than baseline
    patient_ids = np.arange(50)
    baseline_errors = np.random.choice([0, 1, 2], size=50, p=[0.5, 0.3, 0.2])
    # RetinaGuard has fewer errors
    retinaguard_errors = baseline_errors.copy()
    improved_idx = np.random.choice(50, size=15, replace=False)
    retinaguard_errors[improved_idx] = np.maximum(0, retinaguard_errors[improved_idx] - 1)

    diff = baseline_errors - retinaguard_errors
    nonzero_diff = diff[diff != 0]

    stat, p_value = stats.wilcoxon(diff, alternative="greater")

    assert p_value < 0.05  # Statistically significant improvement
    assert stat > 0


def test_permutation_test_significance():
    """Verify non-parametric paired permutation test for difference in metric."""
    np.random.seed(42)
    scores_a = np.array([0.85, 0.88, 0.90, 0.86, 0.89, 0.91, 0.87, 0.89])
    scores_b = np.array([0.75, 0.78, 0.79, 0.74, 0.80, 0.77, 0.76, 0.78])

    observed_diff = np.mean(scores_a) - np.mean(scores_b)

    # Permutation under null hypothesis (no difference)
    n_permutations = 500
    perm_diffs = []
    combined = np.column_stack([scores_a, scores_b])

    for _ in range(n_permutations):
        swaps = np.random.choice([-1, 1], size=len(scores_a))
        perm_diffs.append(np.mean(swaps * (scores_a - scores_b)))

    p_val = np.mean(np.array(perm_diffs) >= observed_diff)
    assert p_val < 0.01  # Highly significant difference


def test_bootstrap_ci_coverage_and_ordering():
    """Verify that bootstrap 95% CI correctly bounds the point estimate."""
    np.random.seed(42)
    y_true = np.random.randint(0, 5, size=80)
    y_pred = y_true.copy()
    # Add noise to 20%
    y_pred[:16] = (y_pred[:16] + 1) % 5

    y_prob = np.zeros((80, 5))
    for i, p in enumerate(y_pred):
        y_prob[i, p] = 0.85
        y_prob[i, (p + 1) % 5] = 0.15

    ci_results = compute_bootstrap_ci(y_true, y_pred, y_prob, n_bootstraps=200, confidence_level=0.95)

    for metric_name in ["quadratic_weighted_kappa", "accuracy", "macro_f1"]:
        stats_dict = ci_results[metric_name]
        mean_val = stats_dict["mean"]
        lower = stats_dict["ci_lower"]
        upper = stats_dict["ci_upper"]

        assert lower <= mean_val <= upper
        assert lower >= -1.0
        assert upper <= 1.0


def test_bootstrap_ci_deterministic_reproducibility():
    """Verify that fixed seed yields identical bootstrap confidence intervals."""
    y_true = np.array([0, 1, 2, 3, 4] * 10)
    y_pred = np.array([0, 1, 2, 3, 3] * 10)

    ci1 = compute_bootstrap_ci(y_true, y_pred, seed=42)
    ci2 = compute_bootstrap_ci(y_true, y_pred, seed=42)

    assert ci1["quadratic_weighted_kappa"] == ci2["quadratic_weighted_kappa"]
    assert ci1["accuracy"] == ci2["accuracy"]
