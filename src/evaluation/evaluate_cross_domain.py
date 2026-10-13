"""RetinaGuard - Cross-Domain Evaluation.

Evaluates DR models across different datasets to measure generalization.

Usage:
    python -m src.evaluation.evaluate_cross_domain --config configs/evaluation.yaml
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline import DRClassifier
from src.training.train_baseline import DRDataset, get_transforms
from src.evaluation.metrics import compute_dr_metrics, save_metrics, format_metrics_report, DR_GRADE_NAMES
from src.utils.logging import get_logger
from src.utils.reproducibility import get_device, set_seed
from src.utils.checkpointing import load_checkpoint

logger = get_logger("evaluate_cross_domain")


class CrossDomainEvaluator:
    """Evaluates model generalization across datasets."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.device = get_device()
        self.results_dir = PROJECT_ROOT / "outputs" / "reports"
        self.figures_dir = PROJECT_ROOT / "outputs" / "figures"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(
        self,
        model: DRClassifier,
        test_df: pd.DataFrame,
        dataset_name: str = "unknown",
        image_size: int = 512,
        batch_size: int = 16,
    ) -> dict[str, Any]:
        """Evaluate a model on a specific dataset.

        Args:
            model: Trained DR classifier.
            test_df: Test metadata DataFrame.
            dataset_name: Name of the evaluation dataset.
            image_size: Input image size.
            batch_size: Batch size for evaluation.

        Returns:
            Evaluation results dictionary.
        """
        model.eval()
        model.to(self.device)

        test_dataset = DRDataset(
            test_df,
            transform=get_transforms(image_size, is_training=False),
        )

        if len(test_dataset) == 0:
            logger.warning(f"No valid samples for {dataset_name}")
            return {"dataset": dataset_name, "error": "no_valid_samples"}

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        )

        all_preds = []
        all_labels = []
        all_probs = []
        all_ids = []

        with torch.no_grad():
            for images, labels, image_ids in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                all_ids.extend(image_ids)

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        metrics = compute_dr_metrics(y_true, y_pred, y_prob)
        metrics["dataset"] = dataset_name
        metrics["timestamp"] = datetime.now().isoformat()

        # Save predictions
        preds_df = pd.DataFrame({
            "image_id": all_ids,
            "true_label": y_true,
            "predicted_label": y_pred,
            **{f"prob_class_{i}": y_prob[:, i] for i in range(y_prob.shape[1])},
        })
        preds_path = self.results_dir / f"predictions_{dataset_name}.csv"
        preds_df.to_csv(preds_path, index=False)

        return metrics

    def run_cross_domain(
        self,
        model: DRClassifier,
        metadata_df: pd.DataFrame,
        source_dataset: str,
        image_size: int = 512,
    ) -> dict[str, dict]:
        """Run cross-domain evaluation across all available datasets.

        Args:
            model: Trained model (trained on source_dataset).
            metadata_df: Full metadata with all datasets.
            source_dataset: Name of the training dataset.
            image_size: Input image size.

        Returns:
            Per-dataset evaluation results.
        """
        results = {}
        datasets = metadata_df["dataset_id"].unique()

        for ds_name in datasets:
            ds_df = metadata_df[
                (metadata_df["dataset_id"] == ds_name) &
                (metadata_df["dr_grade"].notna())
            ]

            if ds_name == source_dataset:
                # Use only test split for source dataset
                ds_df = ds_df[ds_df["split"] == "test"]
                eval_type = "in-domain (test)"
            else:
                eval_type = "cross-domain"

            if len(ds_df) == 0:
                logger.info(f"Skipping {ds_name}: no valid samples")
                continue

            logger.info(f"\nEvaluating on {ds_name} ({eval_type}, n={len(ds_df)})...")
            metrics = self.evaluate_model(model, ds_df, ds_name, image_size)
            metrics["evaluation_type"] = eval_type
            metrics["source_dataset"] = source_dataset
            results[ds_name] = metrics

            logger.info(format_metrics_report(metrics))

        # Generate comparison
        if len(results) > 1:
            self._generate_comparison(results, source_dataset)

        # Save all results
        all_results_path = self.results_dir / "cross_domain_results.json"
        with open(all_results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nSaved cross-domain results: {all_results_path}")

        return results

    def _generate_comparison(self, results: dict, source: str) -> None:
        """Generate comparison tables and plots."""
        # Comparison table
        rows = []
        for ds_name, metrics in results.items():
            rows.append({
                "Dataset": ds_name,
                "Type": metrics.get("evaluation_type", ""),
                "N": metrics.get("n_samples", 0),
                "Accuracy": metrics.get("accuracy", 0),
                "Macro F1": metrics.get("macro_f1", 0),
                "QWK": metrics.get("quadratic_weighted_kappa", 0),
                "AUROC": metrics.get("auroc_macro", None),
            })

        comp_df = pd.DataFrame(rows)
        comp_path = self.results_dir / "cross_domain_comparison.csv"
        comp_df.to_csv(comp_path, index=False)

        logger.info("\nCross-Domain Comparison:")
        logger.info(comp_df.to_string(index=False))

        # Confusion matrix plots
        for ds_name, metrics in results.items():
            if "confusion_matrix" in metrics:
                cm = np.array(metrics["confusion_matrix"])
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[DR_GRADE_NAMES[i] for i in range(len(cm))],
                    yticklabels=[DR_GRADE_NAMES[i] for i in range(len(cm))],
                    ax=ax,
                )
                ax.set_title(f"Confusion Matrix — {ds_name}\n(Source: {source})")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                plt.tight_layout()
                plt.savefig(
                    self.figures_dir / f"confusion_matrix_{ds_name}.png",
                    dpi=150, bbox_inches="tight",
                )
                plt.close()


def main():
    parser = argparse.ArgumentParser(description="RetinaGuard Cross-Domain Evaluation")
    parser.add_argument("--config", type=str, default="configs/evaluation.yaml")
    parser.add_argument("--checkpoint", type=str, required=False,
                       help="Path to model checkpoint")
    parser.add_argument("--metadata", type=str,
                       default="data/metadata/master_metadata.csv")
    parser.add_argument("--source-dataset", type=str, default="APTOS",
                       help="Name of the training dataset")
    args = parser.parse_args()

    set_seed(42)

    metadata_path = PROJECT_ROOT / args.metadata
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        sys.exit(1)

    df = pd.read_csv(metadata_path)

    # Load model
    model = DRClassifier(architecture="resnet18", num_classes=5, pretrained=False)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            load_checkpoint(ckpt_path, model, device=get_device())
            logger.info(f"Loaded checkpoint: {ckpt_path}")
        else:
            logger.error(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)
    else:
        # Try to find latest best model
        exp_dir = PROJECT_ROOT / "experiments" / "baseline"
        if exp_dir.exists():
            exp_dirs = sorted(exp_dir.iterdir(), reverse=True)
            for ed in exp_dirs:
                best = ed / "best_model.pth"
                if best.exists():
                    load_checkpoint(best, model, device=get_device())
                    logger.info(f"Loaded latest checkpoint: {best}")
                    break

    evaluator = CrossDomainEvaluator()
    evaluator.run_cross_domain(model, df, args.source_dataset)


if __name__ == "__main__":
    main()
