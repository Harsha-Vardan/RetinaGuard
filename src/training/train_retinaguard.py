"""RetinaGuard - Training Pipeline for Unified Research Framework.

Trains the RetinaGuard architecture combining:
1. DR Classification (Disease prediction)
2. Lesion Attention Supervision (when lesion masks are present)
3. Domain Classification (on shortcut subspace Z_domain)
4. Domain Adversarial Invariance (via GRL on Z_disease)
5. Statistical Independence Penalty (minimizing Cov(Z_disease, Z_domain))

Usage:
    python -m src.training.train_retinaguard --smoke-test
    python -m src.training.train_retinaguard --epochs 30
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.retinaguard import RetinaGuardModel
from src.models.lesion_guided import compute_lesion_loss
from src.models.causal_disentangler import compute_covariance_penalty
from src.evaluation.metrics import compute_dr_metrics, format_metrics_report
from src.utils.logging import get_logger
from src.utils.reproducibility import set_seed, get_device
from src.utils.checkpointing import save_checkpoint

logger = get_logger("train_retinaguard")


class RetinaGuardDataset(Dataset):
    """PyTorch Dataset supporting images, DR grades, domain IDs, and lesion masks."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        image_size: int = 512,
        is_training: bool = True,
        domain_map: dict | None = None,
    ):
        self.df = metadata.reset_index(drop=True)
        self.image_size = image_size
        self.is_training = is_training
        self.domain_map = domain_map or {"IDRiD": 0, "APTOS": 1}

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])

        # Load fundus image
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        # DR Grade label
        dr_grade = int(row["dr_grade"]) if pd.notna(row["dr_grade"]) else 0

        # Domain label
        dataset_id = str(row.get("dataset_id", "IDRiD"))
        domain_id = self.domain_map.get(dataset_id, 0)

        # Lesion mask (if available)
        mask_path = row.get("lesion_mask_path")
        has_mask = False
        mask_tensor = torch.zeros(1, self.image_size, self.image_size)

        if pd.notna(mask_path) and Path(mask_path).exists():
            try:
                mask_cv = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_cv is not None:
                    mask_cv = cv2.resize(mask_cv, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                    mask_tensor = torch.from_numpy(mask_cv).unsqueeze(0).float() / 255.0
                    has_mask = True
            except Exception:
                has_mask = False

        return {
            "image": image_tensor,
            "dr_grade": torch.tensor(dr_grade, dtype=torch.long),
            "domain_id": torch.tensor(domain_id, dtype=torch.long),
            "mask": mask_tensor,
            "has_mask": torch.tensor(has_mask, dtype=torch.bool),
            "image_id": str(row.get("image_id", img_path.stem)),
        }


class RetinaGuardTrainer:
    """Orchestrates end-to-end training of the RetinaGuard architecture."""

    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 5,
        num_domains: int = 2,
        latent_dim: int = 256,
        lambda_lesion: float = 1.0,
        lambda_domain: float = 0.5,
        lambda_adv: float = 0.25,
        lambda_cov: float = 0.1,
        lr: float = 0.0001,
        weight_decay: float = 0.0001,
        seed: int = 42,
    ):
        set_seed(seed)
        self.device = get_device()
        self.lambda_lesion = lambda_lesion
        self.lambda_domain = lambda_domain
        self.lambda_adv = lambda_adv
        self.lambda_cov = lambda_cov

        self.model = RetinaGuardModel(
            architecture=architecture,
            num_classes=num_classes,
            num_domains=num_domains,
            latent_dim=latent_dim,
            pretrained=True,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.experiment_id = datetime.now().strftime("retinaguard_%Y%m%d_%H%M%S")
        self.experiment_dir = PROJECT_ROOT / "experiments" / "retinaguard" / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized RetinaGuard Framework on {self.device}")
        logger.info(f"Experiment dir: {self.experiment_dir}")

    def train_epoch(self, loader: DataLoader, alpha: float) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        disease_loss_sum = 0.0
        lesion_loss_sum = 0.0
        domain_loss_sum = 0.0
        cov_loss_sum = 0.0

        for batch in loader:
            images = batch["image"].to(self.device)
            dr_grades = batch["dr_grade"].to(self.device)
            domain_ids = batch["domain_id"].to(self.device)
            masks = batch["mask"].to(self.device)
            has_masks = batch["has_mask"].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(images, alpha=alpha)

            # 1. Disease grading loss
            loss_disease = F.cross_entropy(out["disease_logits"], dr_grades)

            # 2. Lesion attention supervision loss
            loss_lesion = compute_lesion_loss(out["attn_logits"], masks, has_masks)

            # 3. Domain classification loss on Z_domain
            loss_domain = F.cross_entropy(out["domain_logits"], domain_ids)

            # 4. Domain adversarial loss on Z_disease
            loss_adv = F.cross_entropy(out["adv_domain_logits"], domain_ids)

            # 5. Independence covariance penalty
            loss_cov = compute_covariance_penalty(out["z_disease"], out["z_domain"])

            # Total joint loss
            loss = (
                loss_disease
                + self.lambda_lesion * loss_lesion
                + self.lambda_domain * loss_domain
                + self.lambda_adv * loss_adv
                + self.lambda_cov * loss_cov
            )

            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            disease_loss_sum += loss_disease.item() * batch_size
            lesion_loss_sum += loss_lesion.item() * batch_size
            domain_loss_sum += loss_domain.item() * batch_size
            cov_loss_sum += loss_cov.item() * batch_size

        n = len(loader.dataset)
        return {
            "total_loss": total_loss / n,
            "disease_loss": disease_loss_sum / n,
            "lesion_loss": lesion_loss_sum / n,
            "domain_loss": domain_loss_sum / n,
            "cov_loss": cov_loss_sum / n,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[dict[str, Any], float]:
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        val_loss = 0.0

        for batch in loader:
            images = batch["image"].to(self.device)
            dr_grades = batch["dr_grade"].to(self.device)

            out = self.model(images, alpha=0.0)
            loss_disease = F.cross_entropy(out["disease_logits"], dr_grades)
            val_loss += loss_disease.item() * images.size(0)

            probs = F.softmax(out["disease_logits"], dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(dr_grades.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        val_loss /= len(loader.dataset)
        metrics = compute_dr_metrics(
            np.array(all_targets), np.array(all_preds), np.array(all_probs), num_classes=5
        )
        return metrics, val_loss

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        epochs: int = 10,
        batch_size: int = 16,
        image_size: int = 512,
    ) -> dict[str, Any]:
        train_ds = RetinaGuardDataset(train_df, image_size=image_size, is_training=True)
        val_ds = RetinaGuardDataset(val_df, image_size=image_size, is_training=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        best_qwk = -1.0
        best_epoch = 0

        logger.info(f"Starting RetinaGuard training: {len(train_ds)} train, {len(val_ds)} val, {epochs} epochs")

        for epoch in range(1, epochs + 1):
            # Dynamic GRL schedule p: alpha ranges 0 to 1
            p = float(epoch - 1) / max(1, epochs)
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

            losses = self.train_epoch(train_loader, alpha=alpha)
            val_metrics, val_loss = self.evaluate(val_loader)

            qwk = val_metrics["quadratic_weighted_kappa"]
            acc = val_metrics["accuracy"]

            logger.info(
                f"Epoch {epoch:2d}/{epochs} | "
                f"Loss: {losses['total_loss']:.4f} (Dis: {losses['disease_loss']:.3f}, Les: {losses['lesion_loss']:.3f}, Cov: {losses['cov_loss']:.3f}) | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f} | Val QWK: {qwk:.4f}"
            )

            if qwk > best_qwk:
                best_qwk = qwk
                best_epoch = epoch
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics,
                    {"architecture": "retinaguard_resnet18"}, self.experiment_dir, "best_retinaguard.pth"
                )
                logger.info(f"  -> Saved new best RetinaGuard model (Val QWK: {qwk:.4f})")

        logger.info(f"Training completed. Best Epoch: {best_epoch}, Best QWK: {best_qwk:.4f}")
        return {"best_epoch": best_epoch, "best_qwk": best_qwk, "experiment_dir": str(self.experiment_dir)}


def main():
    parser = argparse.ArgumentParser(description="Train RetinaGuard Unified Model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    metadata_path = PROJECT_ROOT / "data" / "metadata" / "master_metadata.csv"
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        sys.exit(1)

    df = pd.read_csv(metadata_path)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    epochs = 2 if args.smoke_test else args.epochs
    batch_size = 8 if args.smoke_test else args.batch_size

    if args.smoke_test:
        train_df = train_df.head(24)
        val_df = val_df.head(12)
        logger.info("Executing RetinaGuard Smoke Test Mode")

    trainer = RetinaGuardTrainer()
    trainer.fit(train_df, val_df, epochs=epochs, batch_size=batch_size)


if __name__ == "__main__":
    main()
