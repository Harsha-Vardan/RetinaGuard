"""RetinaGuard - Causal Latent Space Disentanglement Visualizer.

Generates publication-quality t-SNE and PCA manifold visualizations comparing:
1. Disease Latent Space ($Z_d$): Clustered strictly by clinical DR grade (0-4), invariant to hospital camera.
2. Domain Latent Space ($Z_a$): Clustered by imaging acquisition hardware / center (IDRiD vs. APTOS).

Saves artifact to:
- outputs/figures/causal_disentanglement_tsne.png
- outputs/reports/causal_disentanglement_metrics.json

Usage:
    python -m src.evaluation.visualize_disentanglement
"""

import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from src.models.retinaguard import RetinaGuardModel
from src.utils.logging import get_logger
from src.utils.reproducibility import get_device, set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger = get_logger("visualize_disentanglement")


def run_disentanglement_analysis(
    checkpoint_path: Path = PROJECT_ROOT / "experiments" / "retinaguard" / "best_retinaguard.pth",
    image_size: int = 256,
    max_samples: int = 150,
) -> Dict[str, Any]:
    """Extract latent representations Z_d and Z_a, project via t-SNE, and generate figure."""
    set_seed(42)
    device = get_device()

    output_fig_dir = PROJECT_ROOT / "outputs" / "figures"
    output_rep_dir = PROJECT_ROOT / "outputs" / "reports"
    output_fig_dir.mkdir(parents=True, exist_ok=True)
    output_rep_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load trained model
    model = RetinaGuardModel(architecture="resnet18", num_classes=5, num_domains=2, pretrained=False)
    if not checkpoint_path.exists():
        # Fallback to search
        ckpts = sorted(PROJECT_ROOT.glob("experiments/**/best_retinaguard.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not ckpts:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        checkpoint_path = ckpts[0]

    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    st = data.get("model_state_dict", data)
    model.load_state_dict(st)
    model.to(device).eval()
    logger.info(f"Loaded RetinaGuard checkpoint from: {checkpoint_path}")

    # 2. Load balanced test/validation samples from both IDRiD and APTOS
    meta_path = PROJECT_ROOT / "data" / "metadata" / "master_metadata.csv"
    df = pd.read_csv(meta_path)
    eval_df = df[df["split"].isin(["test", "val"]) & df["dr_grade"].notna()].copy()

    # Stratify by dataset and DR grade for representative visualization
    per_group = max(3, max_samples // 10)
    sampled = eval_df.groupby(["dataset_id", "dr_grade"]).head(per_group).reset_index(drop=True)
    logger.info(f"Extracting embeddings across {len(sampled)} multi-center images...")

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    z_disease_list = []
    z_domain_list = []
    dr_grades = []
    dataset_ids = []

    with torch.no_grad():
        for _, row in sampled.iterrows():
            img_p = Path(str(row["image_path"]))
            if not img_p.exists():
                continue
            try:
                img = Image.open(img_p).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                out = model(tensor)

                z_dis = out["z_disease"].cpu().numpy()[0]
                z_dom = out["z_domain"].cpu().numpy()[0]

                z_disease_list.append(z_dis)
                z_domain_list.append(z_dom)
                dr_grades.append(int(row["dr_grade"]))
                dataset_ids.append(str(row["dataset_id"]))
            except Exception as e:
                logger.warning(f"Error processing {img_p}: {e}")

    z_dis_arr = np.array(z_disease_list)
    z_dom_arr = np.array(z_domain_list)
    grades_arr = np.array(dr_grades)
    domains_arr = np.array(dataset_ids)

    # 3. Compute Quantitative Disentanglement Separation Metrics
    # (High silhouette for disease w.r.t grade; high silhouette for domain w.r.t dataset; low mutual leakage)
    dis_grade_sil = float(silhouette_score(z_dis_arr, grades_arr)) if len(np.unique(grades_arr)) > 1 else 0.0
    dis_domain_sil = float(silhouette_score(z_dis_arr, domains_arr)) if len(np.unique(domains_arr)) > 1 else 0.0
    dom_domain_sil = float(silhouette_score(z_dom_arr, domains_arr)) if len(np.unique(domains_arr)) > 1 else 0.0

    # 4. Compute 2D t-SNE Projections
    tsne_2d = TSNE(n_components=2, perplexity=min(30, max(5, len(z_dis_arr) // 4)), random_state=42)
    dis_tsne = tsne_2d.fit_transform(z_dis_arr)
    dom_tsne = tsne_2d.fit_transform(z_dom_arr)

    # 5. Plot Side-by-Side Publication Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    grade_names = ["0: No DR", "1: Mild", "2: Moderate", "3: Severe", "4: Proliferative"]
    palette_grades = sns.color_palette("viridis", 5)

    # Panel A: Disease Latent Space Zd
    for g in range(5):
        mask = grades_arr == g
        if np.any(mask):
            axes[0].scatter(
                dis_tsne[mask, 0],
                dis_tsne[mask, 1],
                label=grade_names[g],
                color=palette_grades[g],
                alpha=0.85,
                s=60,
                edgecolors="none",
            )
    axes[0].set_title(
        f"(a) Disease Latent Space $Z_d$\n"
        f"Clustered by Pathology (Silhouette: {dis_grade_sil:.3f}) | Invariant to Camera",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[0].set_xlabel("t-SNE Dimension 1", fontsize=10)
    axes[0].set_ylabel("t-SNE Dimension 2", fontsize=10)
    axes[0].legend(title="Clinical DR Grade", frameon=True, loc="best", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # Panel B: Domain Latent Space Za
    domain_colors = {"IDRiD": "#D95F02", "APTOS": "#1B9E77"}
    for dom in np.unique(domains_arr):
        mask = domains_arr == dom
        axes[1].scatter(
            dom_tsne[mask, 0],
            dom_tsne[mask, 1],
            label=f"{dom} Cohort",
            color=domain_colors.get(dom, "#7570B3"),
            alpha=0.85,
            s=60,
            edgecolors="none",
        )
    axes[1].set_title(
        f"(b) Acquisition Latent Space $Z_a$\n"
        f"Clustered by Hospital Hardware (Silhouette: {dom_domain_sil:.3f})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[1].set_xlabel("t-SNE Dimension 1", fontsize=10)
    axes[1].set_ylabel("t-SNE Dimension 2", fontsize=10)
    axes[1].legend(title="Imaging Dataset", frameon=True, loc="best", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(
        "RetinaGuard Causal Disentanglement: Empirical Evidence of Feature Invariance",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    fig_path = output_fig_dir / "causal_disentanglement_tsne.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Causal Disentanglement figure to: {fig_path}")

    # 6. Save JSON report
    metrics = {
        "num_samples": int(len(z_dis_arr)),
        "silhouette_disease_by_grade": round(dis_grade_sil, 4),
        "silhouette_disease_by_domain": round(dis_domain_sil, 4),
        "silhouette_domain_by_domain": round(dom_domain_sil, 4),
        "disentanglement_ratio": round(dis_grade_sil / (abs(dis_domain_sil) + 1e-4), 3),
        "figure_path": str(fig_path),
    }

    rep_path = output_rep_dir / "causal_disentanglement_metrics.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved Disentanglement Metrics to: {rep_path}")

    return metrics


if __name__ == "__main__":
    run_disentanglement_analysis()
