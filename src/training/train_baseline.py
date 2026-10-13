"""RetinaGuard - Baseline Training Pipeline.

Trains the baseline DR classification model with configurable
hyperparameters, checkpoint management, and evaluation.

Usage:
    python -m src.training.train_baseline --config configs/baseline.yaml
    python -m src.training.train_baseline --smoke-test
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline import DRClassifier
from src.evaluation.metrics import compute_dr_metrics, save_metrics, format_metrics_report
from src.utils.logging import setup_logger, get_logger
from src.utils.reproducibility import set_seed, get_device, get_hardware_info, save_experiment_config
from src.utils.checkpointing import save_checkpoint, load_checkpoint

logger = get_logger("train_baseline")


class DRDataset(Dataset):
    """PyTorch Dataset for DR classification.

    Loads images from paths specified in a metadata CSV and applies
    configurable transforms.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        image_col: str = "image_path",
        label_col: str = "dr_grade",
        transform: Any = None,
        processed_dir: str | Path | None = None,
    ):
        """Initialize the dataset.

        Args:
            metadata: DataFrame with image paths and labels.
            image_col: Column name for image paths.
            label_col: Column name for DR grade labels.
            transform: torchvision transforms to apply.
            processed_dir: If set, look for processed images here.
        """
        # Filter to records with valid labels
        self.metadata = metadata[metadata[label_col].notna()].reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.processed_dir = Path(processed_dir) if processed_dir else None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """Get a single sample.

        Returns:
            Tuple of (image_tensor, label, image_id).
        """
        row = self.metadata.iloc[idx]
        image_path = row[self.image_col]
        label = int(row[self.label_col])
        image_id = row.get("image_id", str(idx))

        # Try processed dir first
        if self.processed_dir:
            ds_id = row.get("dataset_id", "")
            img_name = Path(image_path).name
            processed_path = self.processed_dir / ds_id / img_name
            if processed_path.exists():
                image_path = str(processed_path)

        # Load image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Return a black image if loading fails
            logger.warning(f"Failed to load {image_path}: {e}")
            img = Image.new("RGB", (512, 512), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, label, image_id

    def get_class_weights(self, num_classes: int = 5) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced training."""
        labels = self.metadata[self.label_col].astype(int).values
        class_counts = np.bincount(labels, minlength=num_classes).astype(float)
        class_counts = np.maximum(class_counts, 1)  # avoid division by zero
        weights = len(labels) / (num_classes * class_counts)
        return torch.FloatTensor(weights)


def get_transforms(image_size: int = 512, is_training: bool = True) -> T.Compose:
    """Get image transforms for training or evaluation.

    Args:
        image_size: Target image size.
        is_training: Whether this is for training (with augmentation).

    Returns:
        Composed transforms.
    """
    if is_training:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class Trainer:
    """Manages baseline model training and evaluation."""

    def __init__(self, config: dict):
        """Initialize trainer with configuration.

        Args:
            config: Training configuration dictionary.
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.train_config = config.get("training", {})

        # Setup
        seed = self.train_config.get("seed", 42)
        set_seed(seed)
        self.device = get_device()
        logger.info(f"Device: {self.device}")
        logger.info(f"Seed: {seed}")

        # Model
        self.model = DRClassifier(
            architecture=self.model_config.get("architecture", "resnet18"),
            num_classes=self.model_config.get("num_classes", 5),
            pretrained=self.model_config.get("pretrained", True),
            dropout=self.model_config.get("dropout", 0.3),
            freeze_backbone=self.model_config.get("freeze_backbone", False),
        ).to(self.device)

        model_info = self.model.get_model_info()
        logger.info(f"Model: {model_info['architecture']}")
        logger.info(f"Parameters: {model_info['total_parameters']:,} "
                    f"(trainable: {model_info['trainable_parameters']:,})")

        # Experiment tracking
        self.experiment_id = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        self.experiment_dir = PROJECT_ROOT / "experiments" / "baseline" / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Best metric tracking
        self.best_metric = -float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run the full training loop.

        Args:
            train_df: Training metadata DataFrame.
            val_df: Validation metadata DataFrame.

        Returns:
            Training results dictionary.
        """
        image_size = self.train_config.get("image_size", [512, 512])
        if isinstance(image_size, list):
            image_size = image_size[0]

        batch_size = self.train_config.get("batch_size", 16)
        num_workers = 0 if self.device.type == "cpu" else min(self.train_config.get("num_workers", 4), 2)
        epochs = self.train_config.get("epochs", 50)
        lr = self.train_config.get("learning_rate", 0.0001)

        # Smoke test mode
        smoke_test = self.train_config.get("smoke_test", {})
        if smoke_test.get("enabled", False):
            max_samples = smoke_test.get("max_samples", 32)
            epochs = smoke_test.get("max_epochs", 2)
            train_df = train_df.head(max_samples)
            val_df = val_df.head(max_samples)
            logger.info(f"SMOKE TEST MODE: {max_samples} samples, {epochs} epochs")

        # Datasets
        train_dataset = DRDataset(
            train_df,
            transform=get_transforms(image_size, is_training=True),
        )
        val_dataset = DRDataset(
            val_df,
            transform=get_transforms(image_size, is_training=False),
        )

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=self.device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=self.device.type == "cuda",
        )

        # Loss function
        class_weights_mode = self.train_config.get("class_weights", "balanced")
        if class_weights_mode == "balanced":
            weights = train_dataset.get_class_weights().to(self.device)
            logger.info(f"Class weights: {weights.tolist()}")
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr,
            weight_decay=self.train_config.get("weight_decay", 0.0001),
        )

        # Scheduler
        scheduler = None
        sched_type = self.train_config.get("scheduler", "cosine")
        if sched_type == "cosine":
            sched_params = self.train_config.get("scheduler_params", {})
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_params.get("T_max", epochs),
                eta_min=sched_params.get("eta_min", 1e-6),
            )

        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_accuracy": [],
                   "val_f1": [], "val_qwk": [], "lr": []}

        logger.info(f"\nStarting training for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self._train_epoch(train_loader, criterion, optimizer)

            # Validate
            val_metrics, val_loss = self._validate(val_loader, criterion)

            # Scheduler step
            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler:
                scheduler.step()

            # Track history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["macro_f1"])
            history["val_qwk"].append(val_metrics["quadratic_weighted_kappa"])
            history["lr"].append(current_lr)

            # Log
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['macro_f1']:.4f} | "
                f"Val QWK: {val_metrics['quadratic_weighted_kappa']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # Checkpoint selection
            monitor = self.train_config.get("early_stopping", {}).get("monitor", "val_qwk")
            current_metric = val_metrics.get(
                monitor.replace("val_", ""),
                val_metrics["quadratic_weighted_kappa"]
            )

            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.patience_counter = 0

                if self.train_config.get("save_best", True):
                    save_checkpoint(
                        self.model, optimizer, epoch, val_metrics, self.config,
                        self.experiment_dir, "best_model.pth", scheduler
                    )
                    logger.info(f"  -> New best model saved ({monitor}: {current_metric:.4f})")
            else:
                self.patience_counter += 1

            # Early stopping
            es_config = self.train_config.get("early_stopping", {})
            if es_config.get("enabled", True):
                patience = es_config.get("patience", 10)
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch} "
                              f"(no improvement for {patience} epochs)")
                    break

        # Save last checkpoint
        if self.train_config.get("save_last", True):
            save_checkpoint(
                self.model, optimizer, epoch, val_metrics, self.config,
                self.experiment_dir, "last_model.pth", scheduler
            )

        elapsed = time.time() - start_time
        logger.info(f"\nTraining complete in {elapsed:.1f}s")
        logger.info(f"Best epoch: {self.best_epoch}, Best {monitor}: {self.best_metric:.4f}")

        # Save training history
        self._save_history(history)

        # Generate plots
        self._plot_training_curves(history)

        # Final evaluation with best model
        best_ckpt = self.experiment_dir / "best_model.pth"
        if best_ckpt.exists():
            load_checkpoint(best_ckpt, self.model, device=self.device)

        final_metrics, _ = self._validate(val_loader, criterion)
        final_metrics["training_time_seconds"] = elapsed
        final_metrics["best_epoch"] = self.best_epoch
        final_metrics["hardware"] = get_hardware_info()

        # Save final metrics
        save_metrics(final_metrics, self.experiment_dir / "final_metrics.json")
        logger.info("\n" + format_metrics_report(final_metrics))

        # Save experiment config
        save_experiment_config(
            self.config, self.experiment_dir, self.experiment_id
        )

        return {
            "experiment_id": self.experiment_id,
            "experiment_dir": str(self.experiment_dir),
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "final_metrics": final_metrics,
            "training_time": elapsed,
        }

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for images, labels, _ in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[dict, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        all_preds = []
        all_labels = []
        all_probs = []
        all_ids = []

        for images, labels, image_ids in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item()
            n_batches += 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_ids.extend(image_ids)

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        metrics = compute_dr_metrics(y_true, y_pred, y_prob)
        avg_loss = total_loss / max(n_batches, 1)

        return metrics, avg_loss

    def _save_history(self, history: dict) -> None:
        """Save training history to CSV and JSON."""
        history_df = pd.DataFrame(history)
        history_df.to_csv(self.experiment_dir / "training_history.csv", index=False)

        with open(self.experiment_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    def _plot_training_curves(self, history: dict) -> None:
        """Generate training curve plots."""
        figures_dir = PROJECT_ROOT / "outputs" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Training Curves — {self.experiment_id}", fontsize=14)

        epochs = range(1, len(history["train_loss"]) + 1)

        # Loss
        axes[0, 0].plot(epochs, history["train_loss"], label="Train", color="#2196F3")
        axes[0, 0].plot(epochs, history["val_loss"], label="Validation", color="#F44336")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy
        axes[0, 1].plot(epochs, history["val_accuracy"], color="#4CAF50")
        axes[0, 1].set_title("Validation Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].grid(True, alpha=0.3)

        # F1 Score
        axes[1, 0].plot(epochs, history["val_f1"], color="#FF9800")
        axes[1, 0].set_title("Validation Macro F1")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].grid(True, alpha=0.3)

        # QWK
        axes[1, 1].plot(epochs, history["val_qwk"], color="#9C27B0")
        axes[1, 1].set_title("Validation QWK")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = figures_dir / f"training_curves_{self.experiment_id}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved training curves: {plot_path}")


def load_config(config_path: str | Path) -> dict:
    """Load training configuration from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="RetinaGuard Baseline Training")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml",
                       help="Path to training config")
    parser.add_argument("--metadata", type=str,
                       default="data/metadata/master_metadata.csv",
                       help="Path to metadata CSV")
    parser.add_argument("--smoke-test", action="store_true",
                       help="Run a quick smoke test")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    metadata_path = PROJECT_ROOT / args.metadata

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    if args.smoke_test:
        config.setdefault("training", {})["smoke_test"] = {
            "enabled": True, "max_samples": 32, "max_epochs": 2,
        }

    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        logger.info("Run 'python -m src.data.build_metadata' first.")
        sys.exit(1)

    # Load metadata
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} records from metadata")

    # Split into train/val
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    if len(train_df) == 0:
        logger.error("No training samples found. Run 'python -m src.data.split_data' first.")
        sys.exit(1)

    logger.info(f"Training: {len(train_df)} samples, Validation: {len(val_df)} samples")

    # Train
    trainer = Trainer(config)
    results = trainer.train(train_df, val_df)

    logger.info(f"\nExperiment complete: {results['experiment_id']}")
    logger.info(f"Results saved to: {results['experiment_dir']}")


if __name__ == "__main__":
    main()
