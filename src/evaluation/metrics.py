"""RetinaGuard - Evaluation Metrics.

Computes classification metrics for DR grading evaluation.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
    classification_report,
)


DR_GRADE_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative",
}


def compute_dr_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    num_classes: int = 5,
) -> dict[str, Any]:
    """Compute comprehensive DR grading metrics.

    Args:
        y_true: Ground truth labels (N,).
        y_pred: Predicted labels (N,).
        y_prob: Predicted probabilities (N, num_classes). Optional.
        num_classes: Number of classes.

    Returns:
        Dictionary of computed metrics.
    """
    metrics = {}

    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Macro F1
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class precision and recall
    present_classes = sorted(set(y_true) | set(y_pred))
    precision = precision_score(
        y_true, y_pred, labels=list(range(num_classes)),
        average=None, zero_division=0
    )
    recall_vals = recall_score(
        y_true, y_pred, labels=list(range(num_classes)),
        average=None, zero_division=0
    )

    metrics["per_class"] = {}
    for i in range(num_classes):
        class_name = DR_GRADE_NAMES.get(i, f"Class {i}")
        class_mask_true = (y_true == i)
        class_mask_pred = (y_pred == i)
        tp = int(np.sum(class_mask_true & class_mask_pred))
        fn = int(np.sum(class_mask_true & ~class_mask_pred))
        fp = int(np.sum(~class_mask_true & class_mask_pred))
        tn = int(np.sum(~class_mask_true & ~class_mask_pred))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

        metrics["per_class"][class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall_vals[i]),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "support": int(np.sum(class_mask_true)),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    metrics["confusion_matrix"] = cm.tolist()

    # Quadratic weighted kappa
    metrics["quadratic_weighted_kappa"] = float(
        cohen_kappa_score(y_true, y_pred, weights="quadratic")
    )

    # AUROC (if probabilities available)
    if y_prob is not None and len(present_classes) > 1:
        try:
            if y_prob.shape[1] == num_classes:
                metrics["auroc_macro"] = float(
                    roc_auc_score(
                        y_true, y_prob, multi_class="ovr",
                        average="macro", labels=list(range(num_classes))
                    )
                )
                # Per-class AUROC
                for i in range(num_classes):
                    if np.sum(y_true == i) > 0:
                        try:
                            binary_true = (y_true == i).astype(int)
                            auc = roc_auc_score(binary_true, y_prob[:, i])
                            class_name = DR_GRADE_NAMES.get(i, f"Class {i}")
                            metrics["per_class"][class_name]["auroc"] = float(auc)
                        except ValueError:
                            pass
        except ValueError as e:
            metrics["auroc_note"] = f"Could not compute AUROC: {e}"

    # Sample count
    metrics["n_samples"] = len(y_true)
    metrics["num_samples"] = len(y_true)
    metrics["class_distribution"] = {
        DR_GRADE_NAMES.get(i, f"Class {i}"): int(np.sum(y_true == i))
        for i in range(num_classes)
    }

    return metrics


def save_metrics(metrics: dict, output_path: str | Path) -> None:
    """Save metrics to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)


def format_metrics_report(metrics: dict, title: str = "DR Classification Metrics Report") -> str:
    """Format metrics as a readable text report."""
    lines = [
        "=" * 50,
        title,
        "=" * 50,
        f"Samples: {metrics.get('n_samples', 'N/A')}",
        f"Accuracy: {metrics.get('accuracy', 0):.4f}",
        f"Macro F1: {metrics.get('macro_f1', 0):.4f}",
        f"QWK: {metrics.get('quadratic_weighted_kappa', 0):.4f}",
    ]

    if "auroc_macro" in metrics:
        lines.append(f"AUROC (macro): {metrics['auroc_macro']:.4f}")

    lines.append("\nPer-class metrics:")
    lines.append(f"{'Class':<15} {'Prec':>6} {'Recall':>6} {'Sens':>6} {'Spec':>6} {'N':>5}")
    lines.append("-" * 50)

    for class_name, class_metrics in metrics.get("per_class", {}).items():
        prec = class_metrics.get("precision", 0)
        rec = class_metrics.get("recall", 0)
        sens = class_metrics.get("sensitivity", 0)
        spec = class_metrics.get("specificity", 0)
        n = class_metrics.get("support", 0)
        lines.append(f"{class_name:<15} {prec:>6.3f} {rec:>6.3f} {sens:>6.3f} {spec:>6.3f} {n:>5}")

    return "\n".join(lines)
