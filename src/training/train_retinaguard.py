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
        lambda_lesion: float = 2.0,
        lambda_domain: float = 0.5,
        lambda_adv: float = 0.2,
        lambda_cov: float = 0.05,
        lr: float = 0.0002,
        weight_decay: float = 0.0001,
        seed: int = 42,
    ):
        set_seed(seed)
        self.device = get_device()
        self.lambda_lesion = lambda_lesion
        self.lambda_domain = lambda_domain
        self.lambda_adv = lambda_adv
        self.lambda_cov = lambda_cov
        self.class_weights: Optional[torch.Tensor] = None

        self.model = RetinaGuardModel(
            architecture=architecture,
            num_classes=num_classes,
            num_domains=num_domains,
            latent_dim=latent_dim,
            pretrained=True,
        ).to(self.device)

        # Discriminative learning rates: backbone adapts smoothly while novel heads learn quickly
        backbone_params = [
            p for n, p in self.model.named_parameters()
            if any(k in n for k in ["stem", "layer1", "layer2", "layer3", "layer4"])
        ]
        head_params = [
            p for n, p in self.model.named_parameters()
            if not any(k in n for k in ["stem", "layer1", "layer2", "layer3", "layer4"])
        ]

        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr * 0.5},
                {"params": head_params, "lr": lr * 2.0},
            ],
            weight_decay=weight_decay,
        )
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        self.experiment_id = datetime.now().strftime("retinaguard_%Y%m%d_%H%M%S")
        self.experiment_dir = PROJECT_ROOT / "experiments" / "retinaguard" / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized RetinaGuard Framework on {self.device}")
        logger.info(f"Experiment dir: {self.experiment_dir}")

    def train_epoch(
        self,
        loader: DataLoader,
        alpha: float,
        cur_lambda_lesion: float,
        cur_lambda_adv: float,
        cur_lambda_cov: float,
        cur_lambda_domain: float,
    ) -> dict[str, float]:
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

            # 1. Disease grading loss with class frequency balance
            loss_disease = F.cross_entropy(
                out["disease_logits"], dr_grades, weight=self.class_weights
            )

            # 2. Lesion attention supervision loss
            loss_lesion = compute_lesion_loss(out["attn_logits"], masks, has_masks)

            # 3. Domain classification loss on Z_domain
            loss_domain = F.cross_entropy(out["domain_logits"], domain_ids)

            # 4. Domain adversarial loss on Z_disease via GRL
            loss_adv = F.cross_entropy(out["adv_domain_logits"], domain_ids)

            # 5. Independence covariance penalty
            loss_cov = compute_covariance_penalty(out["z_disease"], out["z_domain"])

            # Total joint loss with warmup scaling
            loss = (
                loss_disease
                + cur_lambda_lesion * loss_lesion
                + cur_lambda_domain * loss_domain
                + cur_lambda_adv * loss_adv
                + cur_lambda_cov * loss_cov
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
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
            loss_disease = F.cross_entropy(
                out["disease_logits"], dr_grades, weight=self.class_weights
            )
            val_loss += loss_disease.item() * images.size(0)

            probs = F.softmax(out["disease_logits"], dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(dr_grades.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        val_loss /= max(1, len(loader.dataset))
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
        image_size: int = 256,
        warmup_epochs: int = 2,
    ) -> dict[str, Any]:
        # Compute smooth inverse frequency class weights to prevent majority collapse
        grades = train_df["dr_grade"].dropna().astype(int).values
        counts = np.bincount(grades, minlength=5).astype(float)
        counts = np.maximum(counts, 1.0)
        raw_weights = len(grades) / (5.0 * counts)
        smooth_weights = np.sqrt(raw_weights)
        smooth_weights = smooth_weights / smooth_weights.mean()
        self.class_weights = torch.tensor(smooth_weights, dtype=torch.float32, device=self.device)
        logger.info(f"Class-balanced weights: {[round(w, 3) for w in smooth_weights.tolist()]}")

        train_ds = RetinaGuardDataset(train_df, image_size=image_size, is_training=True)
        val_ds = RetinaGuardDataset(val_df, image_size=image_size, is_training=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        best_qwk = -1.0
        best_epoch = 0

        logger.info(f"Starting RetinaGuard training: {len(train_ds)} train, {len(val_ds)} val, {epochs} epochs")

        for epoch in range(1, epochs + 1):
            # Dynamic GRL schedule: alpha ranges 0 to 1
            p = float(epoch - 1) / max(1, epochs)
            alpha = float(2.0 / (1.0 + np.exp(-10 * p)) - 1.0)

            # Adversarial Warmup Schedule: let disease features & lesion attention form first
            if epoch <= warmup_epochs:
                cur_adv = 0.0
                cur_cov = 0.0
                cur_dom = 0.1
                cur_les = self.lambda_lesion
            else:
                frac = float(epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
                cur_adv = self.lambda_adv * frac
                cur_cov = self.lambda_cov * frac
                cur_dom = self.lambda_domain
                cur_les = self.lambda_lesion

            losses = self.train_epoch(
                train_loader,
                alpha=alpha,
                cur_lambda_lesion=cur_les,
                cur_lambda_adv=cur_adv,
                cur_lambda_cov=cur_cov,
                cur_lambda_domain=cur_dom,
            )
            val_metrics, val_loss = self.evaluate(val_loader)
            self.scheduler.step()

            qwk = val_metrics["quadratic_weighted_kappa"]
            acc = val_metrics["accuracy"]

            logger.info(
                f"Epoch {epoch:2d}/{epochs} | "
                f"Loss: {losses['total_loss']:.4f} (Dis: {losses['disease_loss']:.3f}, Les: {losses['lesion_loss']:.3f}, Adv: {losses['cov_loss']:.3f}) | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f} | Val QWK: {qwk:.4f}"
            )

            # Track and save best checkpoint based on QWK (or accuracy if QWK tied)
            is_better = (qwk > best_qwk) or (abs(qwk - best_qwk) < 1e-4 and acc > 0.5)
            if is_better or epoch == 1:
                best_qwk = qwk
                best_epoch = epoch
                # Save to run experiment directory
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics,
                    {"architecture": "retinaguard_resnet18"}, self.experiment_dir, "best_retinaguard.pth"
                )
                # Also save to canonical experiments/retinaguard/ directory for all downstream evaluation
                canonical_dir = PROJECT_ROOT / "experiments" / "retinaguard"
                canonical_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics,
                    {"architecture": "retinaguard_resnet18"}, canonical_dir, "best_retinaguard.pth"
                )
                logger.info(f"  -> Saved new best RetinaGuard model (Val QWK: {qwk:.4f}, Acc: {acc:.4f})")

        logger.info(f"Training completed. Best Epoch: {best_epoch}, Best QWK: {best_qwk:.4f}")
        return {"best_epoch": best_epoch, "best_qwk": best_qwk, "experiment_dir": str(self.experiment_dir)}


def main():
    parser = argparse.ArgumentParser(description="Train RetinaGuard Unified Model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    metadata_path = PROJECT_ROOT / "data" / "metadata" / "master_metadata.csv"
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        sys.exit(1)

    df = pd.read_csv(metadata_path)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    epochs = 2 if args.smoke_test else args.epochs
    batch_size = 8 if args.smoke_test else args.batch_size
    image_size = args.image_size

    if args.smoke_test:
        train_df = train_df.head(24)
        val_df = val_df.head(12)
        logger.info("Executing RetinaGuard Smoke Test Mode")
    elif args.max_samples and len(train_df) > args.max_samples:
        per_class_idrid = max(10, (args.max_samples // 10))
        per_class_aptos = max(10, (args.max_samples // 10))
        train_dfs = []
        for g in range(5):
            idrid_g = train_df[(train_df["dr_grade"] == g) & (train_df["dataset_id"] == "IDRiD")].head(per_class_idrid)
            aptos_g = train_df[(train_df["dr_grade"] == g) & (train_df["dataset_id"] == "APTOS")].head(per_class_aptos)
            train_dfs.extend([idrid_g, aptos_g])
        train_df = pd.concat(train_dfs, ignore_index=True)

        val_dfs = []
        for g in range(5):
            idrid_val = val_df[(val_df["dr_grade"] == g) & (val_df["dataset_id"] == "IDRiD")].head(10)
            aptos_val = val_df[(val_df["dr_grade"] == g) & (val_df["dataset_id"] == "APTOS")].head(10)
            val_dfs.extend([idrid_val, aptos_val])
        val_df = pd.concat(val_dfs, ignore_index=True)
        logger.info(f"Multi-Center Balanced Cohort: {len(train_df)} train ({train_df['dataset_id'].value_counts().to_dict()}), {len(val_df)} val")

    trainer = RetinaGuardTrainer()
    trainer.fit(
        train_df,
        val_df,
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        warmup_epochs=args.warmup_epochs,
    )


if __name__ == "__main__":
    main()
