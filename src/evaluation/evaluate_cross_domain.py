"""RetinaGuard - Cross-Domain Generalization & Leave-One-Center-Out Evaluation.

Evaluates DR models (Baseline ResNet-18 and Unified RetinaGuard) across different
clinical centers (IDRiD and APTOS) to quantify domain shift resistance, out-of-distribution
generalization, and statistical significance via 95% bootstrap confidence intervals.

Usage:
    python -m src.evaluation.evaluate_cross_domain --model-type both --image-size 256
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline import DRClassifier
from src.models.retinaguard import RetinaGuardModel
from src.training.train_baseline import DRDataset, get_transforms
from src.evaluation.metrics import (
    compute_dr_metrics,
    compute_bootstrap_ci,
    save_metrics,
    format_metrics_report,
    DR_GRADE_NAMES,
)
from src.utils.logging import get_logger
from src.utils.reproducibility import get_device, set_seed
from src.utils.checkpointing import load_checkpoint

logger = get_logger("evaluate_cross_domain")


class CrossDomainEvaluator:
    """Evaluates model generalization across clinical datasets and camera centers."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.device = get_device()
        self.results_dir = PROJECT_ROOT / "outputs" / "reports"
        self.figures_dir = PROJECT_ROOT / "outputs" / "figures"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(
        self,
        model: torch.nn.Module,
        test_df: pd.DataFrame,
        dataset_name: str = "unknown",
        image_size: int = 256,
        batch_size: int = 16,
        model_name: str = "model",
    ) -> dict[str, Any]:
        """Evaluate a model on a specific dataset cohort.

        Args:
            model: Trained classifier (DRClassifier or RetinaGuardModel).
            test_df: Test metadata DataFrame.
            dataset_name: Name of the evaluation dataset.
            image_size: Input image resolution.
            batch_size: Evaluation batch size.
            model_name: Identifier for logging/reports.

        Returns:
            Evaluation results dictionary with metrics and 95% bootstrap CIs.
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

                # Unpack logits whether dictionary or direct Tensor
                if isinstance(outputs, dict):
                    logits = outputs["disease_logits"]
                else:
                    logits = outputs

                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                all_ids.extend(image_ids)

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        metrics = compute_dr_metrics(y_true, y_pred, y_prob)
        metrics["dataset"] = dataset_name
        metrics["model_name"] = model_name
        metrics["timestamp"] = datetime.now().isoformat()

        # Compute empirical 95% bootstrap confidence intervals if sufficient samples
        if len(y_true) >= 5:
            try:
                metrics["bootstrap_ci_95"] = compute_bootstrap_ci(
                    y_true, y_pred, y_prob, n_bootstraps=1000, confidence_level=0.95
                )
            except Exception as e:
                logger.warning(f"Could not compute bootstrap CI for {dataset_name}: {e}")

        # Save individual prediction CSV
        preds_df = pd.DataFrame({
            "image_id": all_ids,
            "true_label": y_true,
            "predicted_label": y_pred,
            **{f"prob_class_{i}": y_prob[:, i] for i in range(y_prob.shape[1])},
        })
        safe_name = f"{model_name.lower().replace(' ', '_')}_{dataset_name.lower()}"
        preds_path = self.results_dir / f"predictions_{safe_name}.csv"
        preds_df.to_csv(preds_path, index=False)

        return metrics

    def run_cross_domain(
        self,
        model: torch.nn.Module,
        metadata_df: pd.DataFrame,
        source_dataset: str = "IDRiD",
        image_size: int = 256,
        model_name: str = "RetinaGuard",
    ) -> dict[str, dict]:
        """Run cross-domain evaluation across all available datasets.

        Args:
            model: Trained model.
            metadata_df: Master metadata DataFrame.
            source_dataset: Training / in-domain dataset.
            image_size: Input resolution.
            model_name: Name of model for reporting.

        Returns:
            Per-dataset evaluation results dictionary.
        """
        results = {}
        datasets = sorted(metadata_df["dataset_id"].unique())

        for ds_name in datasets:
            ds_df = metadata_df[
                (metadata_df["dataset_id"] == ds_name) &
                (metadata_df["dr_grade"].notna())
            ]

            if ds_name == source_dataset:
                # Use only test split for source dataset to prevent train leakage
                test_split = ds_df[ds_df["split"] == "test"]
                if len(test_split) > 0:
                    ds_df = test_split
                eval_type = "in-domain (test)"
            else:
                # For unseen domain, evaluate on test or val split if present, else full cohort
                unseen_test = ds_df[ds_df["split"].isin(["test", "val"])]
                if len(unseen_test) > 0:
                    ds_df = unseen_test
                eval_type = "cross-domain"

            if len(ds_df) == 0:
                logger.info(f"Skipping {ds_name}: no valid samples")
                continue

            logger.info(f"\nEvaluating [{model_name}] on {ds_name} ({eval_type}, n={len(ds_df)})...")
            metrics = self.evaluate_model(
                model=model,
                test_df=ds_df,
                dataset_name=ds_name,
                image_size=image_size,
                model_name=model_name,
            )
            metrics["evaluation_type"] = eval_type
            metrics["source_dataset"] = source_dataset
            results[ds_name] = metrics

            logger.info(format_metrics_report(metrics, title=f"{model_name} — {ds_name} ({eval_type})"))

        if len(results) > 1:
            self._generate_comparison(results, source_dataset, model_name)

        return results

    def _generate_comparison(self, results: dict, source: str, model_name: str) -> None:
        """Generate comparison table and confusion matrices."""
        rows = []
        for ds_name, metrics in results.items():
            ci = metrics.get("bootstrap_ci_95", {})
            qwk_ci = ci.get("quadratic_weighted_kappa", {})
            qwk_str = f"{metrics.get('quadratic_weighted_kappa', 0.0):.4f}"
            if qwk_ci:
                qwk_str += f" [{qwk_ci.get('ci_lower', 0.0):.4f}, {qwk_ci.get('ci_upper', 0.0):.4f}]"

            rows.append({
                "Model": model_name,
                "Dataset": ds_name,
                "Type": metrics.get("evaluation_type", ""),
                "N": metrics.get("n_samples", 0),
                "Accuracy": round(metrics.get("accuracy", 0), 4),
                "Macro F1": round(metrics.get("macro_f1", 0), 4),
                "QWK (95% CI)": qwk_str,
                "AUROC": round(metrics.get("auroc_macro", 0.0), 4) if metrics.get("auroc_macro") else "N/A",
            })

        comp_df = pd.DataFrame(rows)
        safe_model = model_name.lower().replace(" ", "_")
        comp_path = self.results_dir / f"cross_domain_{safe_model}.csv"
        comp_df.to_csv(comp_path, index=False)

        logger.info(f"\nCross-Domain Evaluation for {model_name}:")
        logger.info(comp_df.to_string(index=False))

        # Confusion matrix plots
        for ds_name, metrics in results.items():
            if "confusion_matrix" in metrics:
                cm = np.array(metrics["confusion_matrix"])
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[DR_GRADE_NAMES[i] for i in range(len(cm))],
                    yticklabels=[DR_GRADE_NAMES[i] for i in range(len(cm))],
                    ax=ax,
                )
                ax.set_title(f"{model_name} Confusion Matrix — {ds_name}\n(Source: {source})")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                plt.tight_layout()
                plt.savefig(
                    self.figures_dir / f"confusion_matrix_{safe_model}_{ds_name}.png",
                    dpi=150, bbox_inches="tight",
                )
                plt.close()

    def run_comparative_benchmark(
        self,
        baseline_model: torch.nn.Module,
        retinaguard_model: torch.nn.Module,
        metadata_df: pd.DataFrame,
        image_size: int = 256,
        source_dataset: str = "IDRiD",
    ) -> pd.DataFrame:
        """Run side-by-side cross-domain generalization benchmark comparing both models.

        Args:
            baseline_model: Standard ResNet-18 classifier.
            retinaguard_model: Unified RetinaGuard model.
            metadata_df: Metadata DataFrame.
            image_size: Evaluation resolution.
            source_dataset: Primary training center.

        Returns:
            DataFrame summarizing comparative performance and domain retention.
        """
        logger.info("==================================================")
        logger.info("   LEAVE-ONE-CENTER-OUT GENERALIZATION BENCHMARK   ")
        logger.info("==================================================")

        b_results = self.run_cross_domain(
            baseline_model, metadata_df, source_dataset=source_dataset,
            image_size=image_size, model_name="Baseline ResNet-18"
        )

        rg_results = self.run_cross_domain(
            retinaguard_model, metadata_df, source_dataset=source_dataset,
            image_size=image_size, model_name="RetinaGuard Unified"
        )

        rows = []
        for ds_name in sorted(set(b_results.keys()) | set(rg_results.keys())):
            b_m = b_results.get(ds_name, {})
            rg_m = rg_results.get(ds_name, {})

            b_ci = b_m.get("bootstrap_ci_95", {}).get("quadratic_weighted_kappa", {})
            rg_ci = rg_m.get("bootstrap_ci_95", {}).get("quadratic_weighted_kappa", {})

            b_qwk = b_m.get("quadratic_weighted_kappa", 0.0)
            rg_qwk = rg_m.get("quadratic_weighted_kappa", 0.0)

            b_ci_str = f"[{b_ci.get('ci_lower', 0):.3f}, {b_ci.get('ci_upper', 0):.3f}]" if b_ci else "N/A"
            rg_ci_str = f"[{rg_ci.get('ci_lower', 0):.3f}, {rg_ci.get('ci_upper', 0):.3f}]" if rg_ci else "N/A"

            delta_qwk = rg_qwk - b_qwk

            rows.append({
                "Dataset": ds_name,
                "Type": b_m.get("evaluation_type", "cross-domain"),
                "N": b_m.get("n_samples", 0),
                "Baseline QWK": f"{b_qwk:.4f} {b_ci_str}",
                "RetinaGuard QWK": f"{rg_qwk:.4f} {rg_ci_str}",
                "Delta QWK": f"{'+' if delta_qwk >= 0 else ''}{delta_qwk:.4f}",
                "Baseline Acc (%)": f"{b_m.get('accuracy', 0)*100:.2f}%",
                "RetinaGuard Acc (%)": f"{rg_m.get('accuracy', 0)*100:.2f}%",
                "Baseline F1": f"{b_m.get('macro_f1', 0):.4f}",
                "RetinaGuard F1": f"{rg_m.get('macro_f1', 0):.4f}",
            })

        comp_df = pd.DataFrame(rows)
        comp_path = self.results_dir / "cross_domain_comparison.csv"
        comp_df.to_csv(comp_path, index=False)

        json_path = self.results_dir / "cross_domain_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "baseline": b_results,
                "retinaguard": rg_results,
            }, f, indent=2, default=str)

        logger.info("\nComparative Cross-Domain Benchmark Summary:")
        logger.info(comp_df.to_string(index=False))
        logger.info(f"Results saved to: {comp_path} and {json_path}")

        return comp_df


def _load_model_weights(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    """Helper to safely load model state dict."""
    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = data.get("model_state_dict", data)
    model.load_state_dict(state_dict)


def main():
    parser = argparse.ArgumentParser(description="RetinaGuard Cross-Domain Evaluation")
    parser.add_argument("--config", type=str, default="configs/evaluation.yaml")
    parser.add_argument("--model-type", type=str, choices=["both", "retinaguard", "baseline"], default="both",
                        help="Which model(s) to evaluate: 'both', 'retinaguard', or 'baseline'")
    parser.add_argument("--baseline-checkpoint", type=str, default=None,
                        help="Path to Baseline checkpoint")
    parser.add_argument("--retinaguard-checkpoint", type=str, default=None,
                        help="Path to RetinaGuard checkpoint")
    parser.add_argument("--metadata", type=str, default="data/metadata/master_metadata.csv")
    parser.add_argument("--source-dataset", type=str, default="IDRiD",
                        help="Primary training cohort (IDRiD or APTOS)")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Input image resolution")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    metadata_path = PROJECT_ROOT / args.metadata
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        sys.exit(1)

    df = pd.read_csv(metadata_path)
    evaluator = CrossDomainEvaluator()

    # Find checkpoints
    b_path = Path(args.baseline_checkpoint) if args.baseline_checkpoint else None
    rg_path = Path(args.retinaguard_checkpoint) if args.retinaguard_checkpoint else None

    if not b_path:
        ckpts = sorted(
            list(PROJECT_ROOT.glob("experiments/**/best_model.pth")) +
            list(PROJECT_ROOT.glob("experiments/**/best_baseline.pth")),
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        if ckpts:
            b_path = ckpts[0]

    if not rg_path:
        ckpts = sorted(
            list(PROJECT_ROOT.glob("experiments/**/best_retinaguard.pth")),
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        if ckpts:
            rg_path = ckpts[0]

    if args.model_type == "both":
        if not b_path or not b_path.exists():
            logger.error("Baseline checkpoint not found for comparative evaluation.")
            sys.exit(1)
        if not rg_path or not rg_path.exists():
            logger.error("RetinaGuard checkpoint not found for comparative evaluation.")
            sys.exit(1)

        b_model = DRClassifier(architecture="resnet18", num_classes=5, pretrained=False)
        _load_model_weights(b_model, b_path, device)
        logger.info(f"Loaded Baseline checkpoint: {b_path}")

        rg_model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
        _load_model_weights(rg_model, rg_path, device)
        logger.info(f"Loaded RetinaGuard checkpoint: {rg_path}")

        evaluator.run_comparative_benchmark(
            baseline_model=b_model,
            retinaguard_model=rg_model,
            metadata_df=df,
            image_size=args.image_size,
            source_dataset=args.source_dataset,
        )

    elif args.model_type == "retinaguard":
        if not rg_path or not rg_path.exists():
            logger.error("RetinaGuard checkpoint not found.")
            sys.exit(1)
        rg_model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
        _load_model_weights(rg_model, rg_path, device)
        logger.info(f"Loaded RetinaGuard checkpoint: {rg_path}")

        evaluator.run_cross_domain(
            rg_model, df, source_dataset=args.source_dataset,
            image_size=args.image_size, model_name="RetinaGuard Unified"
        )

    elif args.model_type == "baseline":
        if not b_path or not b_path.exists():
            logger.error("Baseline checkpoint not found.")
            sys.exit(1)
        b_model = DRClassifier(architecture="resnet18", num_classes=5, pretrained=False)
        _load_model_weights(b_model, b_path, device)
        logger.info(f"Loaded Baseline checkpoint: {b_path}")

        evaluator.run_cross_domain(
            b_model, df, source_dataset=args.source_dataset,
            image_size=args.image_size, model_name="Baseline ResNet-18"
        )


if __name__ == "__main__":
    main()
