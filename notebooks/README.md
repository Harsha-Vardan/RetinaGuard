# RetinaGuard Research & Experimentation Notebooks

This directory contains the complete Jupyter notebook suite for reproducing all data analysis, model training, ablation studies, clinical explainability, and causal disentanglement manifolds reported in the **RetinaGuard** research paper.

---

## Notebook Suite Overview

| Notebook | Purpose & Key Topics | Interactive Outputs & Figures |
|:---|:---|:---|
| [`01_Data_Exploration_and_Preprocessing.ipynb`](./01_Data_Exploration_and_Preprocessing.ipynb) | Multi-center cohort analysis (IDRiD & APTOS), ICDR DR severity distributions, Circular FOV extraction, LAB CLAHE normalization, and ground-truth lesion masks. | Class distribution plots, raw vs. CLAHE comparisons, pixel-level lesion overlays. |
| [`02_Model_Training_and_Ablations.ipynb`](./02_Model_Training_and_Ablations.ipynb) | End-to-end multi-task training with dynamic GRL adversarial warmup and 4-way component ablation benchmarks. | Trainable parameter audits, loss trajectories, ablation metric comparison bars. |
| [`03_Clinical_Explainability_and_Attribution.ipynb`](./03_Clinical_Explainability_and_Attribution.ipynb) | Interpretability audit: Baseline Grad-CAM vs. RetinaGuard Spatial Lesion Attention, Pointing Game Hit Rate, Attribution IoU, and clinical triage demo. | Side-by-side heatmaps, microaneurysm alignment overlays, benchmark comparison tables. |
| [`04_Causal_Disentanglement_Analysis.ipynb`](./04_Causal_Disentanglement_Analysis.ipynb) | Latent space analysis of disease representations ($Z_d$) vs. acquisition shortcuts ($Z_a$) via t-SNE and silhouette separation scores. | 2-panel publication t-SNE manifold figure, quantitative disentanglement metrics. |
| [`RetinaGuard_Colab_Training.ipynb`](./RetinaGuard_Colab_Training.ipynb) | Standalone 1-click cloud GPU training launcher on Google Colab (free T4 or V100/A100). | Full GPU pipeline execution with dependency installation. |

---

## Recommended Execution Flow

For new users or reviewers reproducing the study from scratch:
1. Start with **`01_Data_Exploration_and_Preprocessing.ipynb`** to inspect the data distribution and preprocessing normalization.
2. Run **`02_Model_Training_and_Ablations.ipynb`** to train the multi-task model or evaluate existing checkpoints.
3. Use **`03_Clinical_Explainability_and_Attribution.ipynb`** to verify that model predictions attend to verified clinical pathology rather than camera shortcuts.
4. Run **`04_Causal_Disentanglement_Analysis.ipynb`** to visualize the causal invariance of the latent space.

---

## Hardware & Environment Requirements
- **Local CPU**: All evaluation, visualization, and inference cells run smoothly on standard multi-core CPUs.
- **GPU Acceleration**: Recommended for training from scratch (`--epochs >= 10`). Compatible with any CUDA-enabled GPU or free Google Colab T4 runtime.
