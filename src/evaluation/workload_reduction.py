"""RetinaGuard - Clinical Workload-Reduction & Health Economics Analysis.

Translates conformal prediction sets into operational clinical metrics:
- Autonomous clearance rate (patients diagnosed without human review)
- Ophthalmologist referral (triage) rate
- Referable-DR sensitivity retained
- Hours/cost saved in screening programmes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.evaluation.calibration import ConformalPredictor

AUTONOMOUS_CLASSES_DEFAULT = (0, 4)  # clear Normal or clear Proliferative


def _to_arrays(y_true, y_prob):
    if y_prob is None:
        return None, None
    yp = np.asarray(y_prob, dtype=float)
    yt = None if y_true is None else np.asarray(y_true).astype(int).ravel()
    return yt, yp


def compute_prediction_sets(
    y_prob: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    alpha: float = 0.10,
    calibrate_on_same: bool = True,
) -> List[List[int]]:
    """Generate conformal prediction sets for a cohort."""
    probs = np.asarray(y_prob, dtype=float)
    cp = ConformalPredictor(alpha=alpha)
    if y_true is not None and calibrate_on_same:
        try:
            cp.calibrate(probs, np.asarray(y_true).astype(int).ravel())
        except Exception:
            pass
    else:
        # fallback threshold
        cp.q_hat = 1.0 - (1.0 - alpha)
        cp.calibrated = True
    sets = [cp.predict_set(probs[i]) for i in range(len(probs))]
    return sets


def compute_workload_reduction(
    y_true: Optional[Union[np.ndarray, Sequence[int]]] = None,
    y_prob: Optional[Union[np.ndarray, Sequence]] = None,
    prediction_sets: Optional[Sequence[Sequence[int]]] = None,
    alpha: float = 0.10,
    autonomous_classes: Sequence[int] = AUTONOMOUS_CLASSES_DEFAULT,
    minutes_per_case: float = 5.0,
    cost_per_hour: float = 60.0,
    calibrate_on_same: bool = True,
) -> Dict[str, Any]:
    """Compute workload-reduction / health-economics metrics.

    Autonomous clearance is granted when the conformal set is a singleton whose
    single class is in ``autonomous_classes`` (default {0, 4}: clear Normal or
    clear Proliferative DR). All other cases are triaged to a specialist.
    """
    auto_set = set(int(c) for c in autonomous_classes)

    if prediction_sets is None:
        if y_prob is None:
            raise ValueError("Either prediction_sets or y_prob must be provided.")
        prediction_sets = compute_prediction_sets(y_prob, y_true=y_true, alpha=alpha, calibrate_on_same=calibrate_on_same)

    sets = [list(s) for s in prediction_sets]
    n = len(sets)
    if n == 0:
        return {
            "n_samples": 0,
            "autonomous_clearance_rate": 0.0,
            "autonomous_rate": 0.0,
            "referral_rate": 1.0,
            "triage_rate": 1.0,
            "ophthalmologist_referral_rate": 1.0,
            "mean_set_size": 0.0,
            "singleton_rate": 0.0,
            "ambiguous_triage_rate": 0.0,
            "referable_sensitivity": 1.0,
            "hours_saved": 0.0,
            "cost_saved": 0.0,
        }

    autonomous_mask = np.array([ (len(s) == 1 and int(s[0]) in auto_set) for s in sets ], dtype=bool)
    autonomous_rate = float(np.mean(autonomous_mask))
    referral_rate = 1.0 - autonomous_rate

    set_sizes = np.array([len(s) for s in sets])
    singleton_rate = float(np.mean(set_sizes == 1))
    ambiguous_rate = float(np.mean(set_sizes >= 3))

    # Referable sensitivity: fraction of truly referable (>=2) cases NOT autonomously cleared as non-referable
    referable_sensitivity = 1.0
    if y_true is not None:
        yt = np.asarray(y_true).astype(int).ravel()
        referable = yt >= 2
        if referable.any():
            # dangerous: referable case autonomously cleared as class 0 (non-referable)
            dangerous = autonomous_mask & referable & np.array([ (len(s)==1 and s[0] == 0) for s in sets ])
            referable_sensitivity = float(1.0 - (dangerous.sum() / referable.sum()))

    minutes_saved = autonomous_rate * n * minutes_per_case
    hours_saved = minutes_saved / 60.0
    cost_saved = hours_saved * cost_per_hour

    return {
        "n_samples": n,
        "autonomous_clearance_rate": round(autonomous_rate, 4),
        "autonomous_rate": round(autonomous_rate, 4),
        "referral_rate": round(referral_rate, 4),
        "triage_rate": round(referral_rate, 4),
        "ophthalmologist_referral_rate": round(referral_rate, 4),
        "mean_set_size": round(float(np.mean(set_sizes)), 4),
        "singleton_rate": round(singleton_rate, 4),
        "ambiguous_triage_rate": round(ambiguous_rate, 4),
        "referable_sensitivity": round(referable_sensitivity, 4),
        "hours_saved": round(hours_saved, 4),
        "cost_saved": round(cost_saved, 4),
        "minutes_saved": round(minutes_saved, 4),
    }


def workload_reduction_curve(
    y_true: Optional[Union[np.ndarray, Sequence[int]]] = None,
    y_prob: Optional[Union[np.ndarray, Sequence]] = None,
    prediction_sets_list: Optional[Sequence[Sequence[Sequence[int]]]] = None,
    alphas: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
    autonomous_classes: Sequence[int] = AUTONOMOUS_CLASSES_DEFAULT,
    minutes_per_case: float = 5.0,
) -> List[Dict[str, Any]]:
    """Compute workload-reduction metrics across a range of alpha (risk) levels."""
    results = []
    if prediction_sets_list is not None:
        for i, alpha in enumerate(alphas):
            sets = prediction_sets_list[i] if i < len(prediction_sets_list) else prediction_sets_list[-1]
            m = compute_workload_reduction(
                y_true=y_true, prediction_sets=sets, alpha=alpha,
                autonomous_classes=autonomous_classes, minutes_per_case=minutes_per_case,
            )
            m["alpha"] = float(alpha)
            m["target_coverage"] = round(1.0 - float(alpha), 4)
            results.append(m)
        return results

    for alpha in alphas:
        m = compute_workload_reduction(
            y_true=y_true, y_prob=y_prob, alpha=float(alpha),
            autonomous_classes=autonomous_classes, minutes_per_case=minutes_per_case,
        )
        m["alpha"] = float(alpha)
        m["target_coverage"] = round(1.0 - float(alpha), 4)
        results.append(m)
    return results


def health_economics_analysis(
    y_true=None,
    y_prob=None,
    prediction_sets=None,
    alpha: float = 0.10,
    annual_patients: int = 10000,
    minutes_per_case: float = 5.0,
    cost_per_hour: float = 60.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """High-level health-economics summary for a screening programme."""
    m = compute_workload_reduction(
        y_true=y_true, y_prob=y_prob, prediction_sets=prediction_sets,
        alpha=alpha, minutes_per_case=minutes_per_case, cost_per_hour=cost_per_hour, **kwargs,
    )
    annual_minutes_saved = m["autonomous_rate"] * annual_patients * minutes_per_case
    return {
        **m,
        "annual_patients": annual_patients,
        "annual_minutes_saved": round(annual_minutes_saved, 2),
        "annual_hours_saved": round(annual_minutes_saved / 60.0, 2),
        "annual_cost_saved": round((annual_minutes_saved / 60.0) * cost_per_hour, 2),
    }


# Aliases
compute_workload_reduction_curve = workload_reduction_curve
workload_curve = workload_reduction_curve
health_economics = health_economics_analysis
clinical_workload_reduction = compute_workload_reduction
workload_reduction = compute_workload_reduction
triage_analysis = compute_workload_reduction
screening_workload = compute_workload_reduction
