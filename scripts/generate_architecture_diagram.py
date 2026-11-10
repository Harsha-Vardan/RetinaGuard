"""RetinaGuard - Publication-Quality Architectural Diagram Generator.

Generates a comprehensive, clean, vector-style architectural diagram showing:
1. Input Fundus Photograph & Circular FOV + LAB CLAHE Preprocessing
2. Hierarchical Convolutional Backbone (ResNet / ConvNeXt / EfficientNet)
3. Spatial Lesion Attention Decoder with Auxiliary Supervision
4. Dual-Path Feature Pooling (Global Average Pooling + Lesion Attention Pooling)
5. Causal Disentanglement Projections (Z_disease vs Z_domain)
6. Gradient Reversal Layer (GRL) & Adversarial Invariance Branch
7. Hilbert-Schmidt Cross-Covariance Independence Constraint
8. Clinical DR Grading Head & Conformal Prediction Risk-Controlled Triage Sets
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def draw_architecture_diagram():
    fig, ax = plt.subplots(figsize=(19, 10), dpi=300)
    ax.set_xlim(0, 190)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Background color
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0b0f19")

    # Helper styling function
    def draw_box(x, y, w, h, title, subtitle="", color="#1e293b", edge="#38bdf8", lw=1.8, rx=2.0):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.5,rounding_size={rx}",
            facecolor=color, edgecolor=edge, linewidth=lw
        )
        ax.add_patch(rect)
        if title:
            ax.text(x + w / 2, y + h / 2 + (1.8 if subtitle else 0), title,
                    ha="center", va="center", color="#f8fafc", fontsize=10, fontweight="bold")
        if subtitle:
            ax.text(x + w / 2, y + h / 2 - 2.2, subtitle,
                    ha="center", va="center", color="#94a3b8", fontsize=8)

    def draw_arrow(x1, y1, x2, y2, color="#38bdf8", lw=1.8, label="", label_offset=(0, 2.5), style="->"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=2, shrinkB=2)
        )
        if label:
            ax.text((x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1],
                    label, ha="center", va="center", color=color, fontsize=8, fontweight="bold")

    # Header Title Banner
    ax.text(95, 96, "RetinaGuard: Lesion-Guided, Causally Disentangled Diagnostic Architecture",
            ha="center", va="center", color="#38bdf8", fontsize=15, fontweight="heavy", fontfamily="sans-serif")
    ax.text(95, 92.5, "Structural Causal Model (SCM) • Spatial Lesion Attention • Split Conformal Clinical Triage",
            ha="center", va="center", color="#94a3b8", fontsize=9.5)

    # -------------------------------------------------------------
    # 1. INPUT & PREPROCESSING ZONE (x: 5 - 32)
    # -------------------------------------------------------------
    draw_box(4, 50, 25, 34, "1. Retinal Input Stage", "Raw Multi-Center Fundus Photograph\n(516x516x3)",
             color="#131d33", edge="#60a5fa", lw=1.5)
    draw_box(6, 68, 21, 10, "Raw Fundus Image X", "Variable Optics & Cameras",
             color="#1e293b", edge="#94a3b8", lw=1.2)
    draw_box(6, 53, 21, 11, "Retinal Preprocessor", "Circular FOV Crop + CLAHE\nStandardized (512x512)",
             color="#1e293b", edge="#38bdf8", lw=1.5)
    draw_arrow(16.5, 68, 16.5, 64, color="#60a5fa")

    # Arrow to Backbone
    draw_arrow(29, 58.5, 35, 58.5, color="#38bdf8", label="X_prep", lw=2)

    # -------------------------------------------------------------
    # 2. CONVOLUTIONAL BACKBONE ZONE (x: 35 - 62)
    # -------------------------------------------------------------
    draw_box(35, 45, 27, 39, "2. Feature Extraction Backbone", "Multi-Scale Convolutional Feature Maps",
             color="#111c38", edge="#818cf8", lw=1.5)
    draw_box(37, 68, 23, 11, "Conv Backbone (Layer 1-3)", "ResNet / ConvNeXt / EfficientNet\nLow-to-Mid Level Features",
             color="#1e293b", edge="#818cf8", lw=1.2)
    draw_box(37, 49, 23, 13, "Backbone Layer 4", "Deep Spatial Semantics\nF in R^(C x H' x W') (512x16x16)",
             color="#241442", edge="#a855f7", lw=1.8)
    draw_arrow(48.5, 68, 48.5, 62, color="#818cf8")

    # -------------------------------------------------------------
    # 3. SPATIAL LESION ATTENTION DECODER (x: 70 - 100, upper branch)
    # -------------------------------------------------------------
    draw_box(69, 58, 32, 28, "3. Spatial Lesion Attention Decoder", "Guided by Pathological Biomarkers",
             color="#1f1538", edge="#c084fc", lw=1.8)
    draw_box(71, 71, 28, 10, "Attention Conv Blocks", "3x3 Conv (dim/4) -> BN -> ReLU\n1x1 Conv -> Sigmoid",
             color="#2a1b4e", edge="#c084fc", lw=1.2)
    draw_box(71, 60, 28, 9, "Predicted Mask M^", "Spatial Probability Map (16x16)",
             color="#3b1d60", edge="#e879f9", lw=1.5)
    draw_arrow(85, 71, 85, 69, color="#c084fc")

    # Branch from Layer 4 to Lesion Attention
    draw_arrow(60, 58, 69, 73, color="#c084fc", label="F", lw=1.8)

    # Auxiliary Supervision Box (Above Lesion Attention)
    draw_box(71, 38, 28, 14, "Auxiliary Lesion Loss L_lesion", "Supervised on 444 Doctor Masks\n(MA + HE + EX + SE)",
             color="#3f1322", edge="#f43f5e", lw=1.8)
    draw_arrow(85, 60, 85, 52, color="#f43f5e", label="BCE Loss", style="<->")

    # -------------------------------------------------------------
    # 4. DUAL-PATH FEATURE FUSION (x: 106 - 130)
    # -------------------------------------------------------------
    draw_box(106, 50, 25, 36, "4. Dual-Path Feature Fusion", "Pathology + Context Integration",
             color="#0f2b38", edge="#2dd4bf", lw=1.8)
    draw_box(108, 72, 21, 9, "Global Avg Pool (GAP)", "z_global in R^512",
             color="#133e4d", edge="#2dd4bf", lw=1.2)
    draw_box(108, 60, 21, 9, "Lesion Attn Pool", "z_lesion in R^512",
             color="#133e4d", edge="#2dd4bf", lw=1.2)
    draw_box(108, 52, 21, 6, "Fused Embedding", "z_fused in R^1024",
             color="#134e4a", edge="#14b8a6", lw=1.5)

    # Connect Layer 4 to GAP directly
    draw_arrow(60, 52, 108, 76, color="#2dd4bf", label="Global Path", lw=1.5)
    # Connect Attention Mask + F to Lesion Attn Pool
    draw_arrow(99, 64, 108, 64, color="#e879f9", label="M^ * F", lw=1.5)

    # -------------------------------------------------------------
    # 5. CAUSAL DISENTANGLEMENT & GRL (x: 136 - 165)
    # -------------------------------------------------------------
    draw_box(136, 42, 28, 48, "5. Causal Disentanglement", "Structural Separation Z_dis _|_ Z_dom",
             color="#18243b", edge="#38bdf8", lw=1.8)
    
    # Disease Subspace
    draw_box(138, 73, 24, 12, "Disease Latent Z_disease", "Pure Biological Pathology\nZ_dis in R^256",
             color="#0c3d2e", edge="#10b981", lw=1.8)

    # Domain Subspace
    draw_box(138, 46, 24, 12, "Domain Latent Z_domain", "Camera / Lighting Shortcuts\nZ_dom in R^256",
             color="#422006", edge="#f59e0b", lw=1.8)

    # Covariance Penalty Link
    ax.annotate(
        "Cov(Z_dis, Z_dom) -> 0\nHilbert-Schmidt Penalty L_cov",
        xy=(150, 73), xytext=(150, 58),
        ha="center", va="center", color="#f59e0b", fontsize=7.5, fontweight="bold",
        arrowprops=dict(arrowstyle="<->", color="#f59e0b", lw=1.5, ls="--")
    )

    # Arrow from Fused to Projections
    draw_arrow(129, 55, 138, 78, color="#10b981", label="Proj_dis", lw=1.6)
    draw_arrow(129, 53, 138, 52, color="#f59e0b", label="Proj_dom", lw=1.6)

    # -------------------------------------------------------------
    # 6. OUTPUTS, GRL ADVERSARY & CONFORMAL TRIAGE (x: 168 - 188)
    # -------------------------------------------------------------
    # Adversarial Branch (GRL)
    draw_box(167, 34, 21, 13, "Domain Adversary", "GRL (Grad Reverse -lambda)\nPenalizes Camera Leaks L_adv",
             color="#3b1320", edge="#ef4444", lw=1.5)
    draw_arrow(162, 75, 167, 43, color="#ef4444", label="GRL (-alpha)", lw=1.5)

    # Domain Classifier
    draw_box(167, 50, 21, 10, "Domain Classifier", "Hospital ID (d^ in 1..K)\nLoss L_domain",
             color="#332007", edge="#f59e0b", lw=1.2)
    draw_arrow(162, 52, 167, 54, color="#f59e0b")

    # Disease Grading Head
    draw_box(167, 72, 21, 13, "DR Classifier Head", "ICDR Grades 0 to 4\nAsymmetric Cost Loss L_cost",
             color="#064e3b", edge="#34d399", lw=2.0)
    draw_arrow(162, 78, 167, 78, color="#34d399", label="Softmax", lw=1.8)

    # Conformal Risk-Controlled Triage Box (Final Stage)
    draw_box(136, 6, 52, 23, "6. Conformal Risk-Controlled Clinical Triage",
             "Finite-Sample Safety Bound: P(Y in C(X)) >= 1 - alpha (90% Guaranteed Coverage)",
             color="#1e1b4b", edge="#a855f7", lw=2.0)

    draw_box(139, 9, 21, 13, "Singleton Set {Grade k}", "High Confidence\nAutonomous Clearance\n(Safe for Screening)",
             color="#064e3b", edge="#10b981", lw=1.5)
    draw_box(164, 9, 21, 13, "Multi-Grade Set {0, 1}", "Borderline Ambiguity\nSpecialist Triage Referral\n(Zero Blind Guessing)",
             color="#451a03", edge="#f97316", lw=1.5)

    # Arrow from Classifier to Conformal Box
    draw_arrow(177.5, 72, 177.5, 30, color="#a855f7", label="Probabilities P(y|X)", lw=1.8)

    # -------------------------------------------------------------
    # Bottom Mathematical Formulation Summary Bar
    # -------------------------------------------------------------
    summary_box = patches.FancyBboxPatch(
        (4, 6), 127, 23,
        boxstyle="round,pad=0.5,rounding_size=2.0",
        facecolor="#0f172a", edgecolor="#334155", linewidth=1.2
    )
    ax.add_patch(summary_box)

    ax.text(67.5, 25, "Unified Multi-Objective Loss Function:",
            ha="center", va="center", color="#38bdf8", fontsize=9.5, fontweight="bold")
    ax.text(67.5, 19.5,
            r"$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{disease}} + \lambda_1 \mathcal{L}_{\mathrm{lesion}} + \lambda_2 \mathcal{L}_{\mathrm{domain}} + \lambda_3 \mathcal{L}_{\mathrm{adv}} + \lambda_4 \mathcal{L}_{\mathrm{cov}} + \lambda_5 \mathcal{L}_{\mathrm{cost}}$",
            ha="center", va="center", color="#f8fafc", fontsize=9)
    ax.text(67.5, 12,
            "Key Clinical Guarantee: Structural independence Z_disease _|_ Z_domain ensures diagnostic invariance to camera manufacturer;\n"
            "Auxiliary lesion attention anchors representations to microaneurysms; Conformal triage prevents dangerous false-negative misdiagnosis.",
            ha="center", va="center", color="#94a3b8", fontsize=7.8)

    plt.tight_layout()
    out_path = FIGURES_DIR / "retinaguard_architecture_diagram.png"
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Successfully generated Architectural Diagram: {out_path}")


if __name__ == "__main__":
    draw_architecture_diagram()
