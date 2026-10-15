# RetinaGuard: Lesion-Guided, Causally Informed Diabetic Retinopathy Diagnosis

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-110%2F110%20Passed-brightgreen?logo=pytest)](file:///D:/Mini-project/RetinaGuard/tests)
[![Clinical Cohorts](https://img.shields.io/badge/Evaluated%20Cohorts-IDRiD%20%28Lesions%29%20%2B%20APTOS%20%28Transfer%29-orange)](file:///D:/Mini-project/RetinaGuard/data)
[![License](https://img.shields.io/badge/License-MIT%20%2F%20Research-purple)](#license)
[![Status](https://img.shields.io/badge/Status-Active%20Research-success)]()

**A deep representation learning framework that separates disease pathology from acquisition confounders to enable robust, trustworthy, and clinically generalizable Diabetic Retinopathy grading.**

[Key Innovations](#key-innovations) •
[Architecture](#architecture-overview) •
[Clinical Datasets](#clinical-datasets) •
[Getting Started](#getting-started) •
[CLI Reference](#cli-reference) •
[Evaluation](#cross-domain-evaluation) •
[Test Suite](#verification--test-suite)

</div>

---

> [!IMPORTANT]
> **Clinical Disclaimer**: RetinaGuard is an experimental medical AI research project developed for academic evaluation. It is **not** an FDA/CE-cleared medical device and must **not** be used as a standalone diagnostic system in clinical healthcare practice.

---

## The Research Problem: Shortcut Learning in DR

Deep learning models for Diabetic Retinopathy (DR) grading frequently report high AUC on homogeneous benchmark test splits. However, in real-world clinical deployments across hospitals, different fundus cameras, or diverse populations, performance significantly degrades.

Models often exploit **acquisition shortcuts**—such as illumination variations, camera sensor artifacts, field-of-view vignetting, or color temperature—instead of learning true retinal pathology:

```
Acquisition Artifacts (Camera/Hospital) ───[Spurious Shortcut]───► High Benchmark Accuracy
                                                                          │
                                                                   (Cross-Site Failure!)
True Pathology (Microaneurysms, Exudates) ──[Causal Feature]─────► True Clinical Generalization
```

### Core Research Questions
1. **Cross-Domain Generalization**: How severely does DR grading degrade when evaluated on unseen hospital cohorts?
2. **Lesion Guidance**: Does explicit spatial attention supervision on retinal lesions prevent background shortcut learning?
3. **Causal Disentanglement**: Can disease representations ($Z_d$) be statistically separated from acquisition/site representations ($Z_a$)?
4. **Attribution Faithfulness**: Do feature attributions align with verified clinical lesion segmentations?

---

## Key Innovations

RetinaGuard addresses shortcut learning through a unified, multi-task architecture:

1. **Spatial Lesion-Guided Attention**:
   - Dynamically learns a spatial attention map $\mathcal{M} \in [0, 1]^{H \times W}$ supervised by ground-truth lesion segmentations (microaneurysms, hemorrhages, hard/soft exudates).
   - Focuses deep convolutional representations on genuine pathological manifestations before global pooling.
2. **Causal Representation Disentangler**:
   - Decomposes the latent representation into:
     - **Disease Representation ($Z_d \in \mathbb{R}^{256}$)**: Encodes true DR progression (grades 0 to 4).
     - **Acquisition Representation ($Z_a \in \mathbb{R}^{128}$)**: Captures hospital, camera, and device characteristics.
   - **Adversarial Domain Invariance**: Applies a Gradient Reversal Layer (GRL) on $Z_d$ to ensure $Z_d$ contains zero site-predictive mutual information.
   - **Cross-Covariance Independence Penalty**: Minimizes the Hilbert-Schmidt cross-covariance between $Z_d$ and $Z_a$, enforcing statistical independence:
     $$\mathcal{L}_{\text{cov}}(Z_d, Z_a) = \frac{1}{d_d \cdot d_a} \left\| (Z_d - \bar{Z}_d)^\top (Z_a - \bar{Z}_a) \right\|_F^2$$
3. **Robust Clinical Preprocessing**:
   - Automated Retinal Field-of-View (FOV) detection and black border cropping.
   - Contrast Limited Adaptive Histogram Equalization (CLAHE) in CIE L\*a\*b\* color space.
   - Aspect-ratio preserving letterboxing with circular aperture masking.

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │   Input Fundus Image X (3 x 512 x 512) │
                      └──────────────────┬─────────────────────┘
                                         ▼
                      ┌────────────────────────────────────────┐
                      │     Feature Backbone (ResNet / CNN)    │
                      └──────────────────┬─────────────────────┘
                                         ▼
                      ┌────────────────────────────────────────┐
                      │   Deep Feature Map F (C x H x W)       │
                      └──────────┬───────────────────┬─────────┘
                                 │                   │
                                 ▼                   │
            ┌─────────────────────────────┐          │
            │   Spatial Lesion Attention  │          │
            └────────────┬────────────────┘          │
                         │                           │
              Auxiliary  ▼  Supervision              │
     [Lesion Loss: BCE / Focal with GT]              ▼
                         │               ┌───────────────────────┐
                         └──────────────►│ Gated Feature Fusion  │
                                         └───────────┬───────────┘
                                                     ▼
                                         ┌───────────────────────┐
                                         │  Global Pooling (GAP) │
                                         └───────────┬───────────┘
                                                     ▼
                                         ┌───────────────────────┐
                                         │  Causal Disentangler  │
                                         └─────┬───────────┬─────┘
                                               │           │
                     ┌─────────────────────────┘           └────────────────────────┐
                     ▼                                                              ▼
       ┌───────────────────────────┐                                  ┌───────────────────────────┐
       │   Disease Latent (Z_d)    │◄──────[ Covariance Penalty ]────►│  Acquisition Latent (Z_a) │
       └───────┬───────────┬───────┘                                  └─────────────┬─────────────┘
               │           │                                                        │
               ▼           ▼ (Gradient Reversal)                                    ▼
       ┌───────────────┐ ┌─────────────────────────┐                  ┌───────────────────────────┐
       │ DR Classifier │ │  Domain Classifier      │                  │ Domain Classifier         │
       │ (Grades 0-4)  │ │  (Adversarial Minimax)  │                  │ (Hospital / Device ID)    │
       └───────────────┘ └─────────────────────────┘                  └───────────────────────────┘
```

---

## Project Structure

```
RetinaGuard/
├── configs/                          # Experiment configuration YAMLs
│   ├── baseline.yaml                 # Baseline model hyperparameters
│   ├── datasets.yaml                 # Multi-dataset registry and download protocols
│   ├── evaluation.yaml               # Cross-domain validation protocol
│   └── preprocessing.yaml            # FOV crop, CLAHE, and image normalization
├── data/
│   ├── raw/                          # Raw datasets (IDRiD, APTOS, etc.)
│   ├── processed/                    # Standardized 512x512 preprocessed images
│   ├── manifests/                    # Patient-stratified split manifests (train/val/test)
│   └── metadata/                     # master_metadata.csv and dataset audit registry
├── docs/                             # Engineering and clinical documentation
│   ├── DATASET_ACQUISITION_REPORT.md # Download status and integrity checksums
│   ├── DATASET_AUDIT.md              # Quality metrics, blur, contrast, and label counts
│   ├── DATASET_SCHEMA.md             # Unified 17-column master metadata specification
│   └── IMPLEMENTATION_STATUS.md      # Phase-by-phase verification logs
├── experiments/                      # Checkpoints, metric logs, and training artifacts
├── outputs/                          # Evaluation plots, confusion matrices, and figures
├── src/
│   ├── data/
│   │   ├── audit_dataset.py          # Data integrity audit & corruption detector
│   │   ├── build_metadata.py         # Multi-dataset ingestion into master schema
│   │   ├── create_mock_data.py       # High-fidelity synthetic fundus & mask generator
│   │   ├── download_datasets.py      # Automated dataset downloader
│   │   ├── preprocessing.py          # FOV extraction, CLAHE, letterbox resize
│   │   └── split_data.py             # Stratified patient-aware splitting
│   ├── evaluation/
│   │   ├── evaluate_cross_domain.py  # Cross-domain generalizability evaluation
│   │   └── metrics.py                # QWK, Macro-AUC, sensitivity, specificity, ECE
│   ├── inference/
│   │   └── predict.py                # Single-image and batch inference CLI
│   ├── models/
│   │   ├── baseline.py               # Standard ResNet / ConvNeXt classifiers
│   │   ├── lesion_guided.py          # Spatial Lesion Attention & auxiliary supervision
│   │   ├── causal_disentangler.py    # Disentangled representations & GRL domain adversary
│   │   └── retinaguard.py            # Unified end-to-end multi-task framework
│   ├── training/
│   │   ├── train_baseline.py         # Standard baseline training loop
│   │   └── train_retinaguard.py      # Multi-task research training pipeline
│   └── utils/
│       ├── checkpointing.py          # Atomic model save/restore utilities
│       ├── logging.py                # UTF-8 & Windows CP1252-safe logging
│       └── reproducibility.py        # Seed management and hardware profiler
├── tests/                            # Automated test suite (21 unit tests)
├── pyproject.toml                    # Build configuration and pytest flags
├── requirements.txt                  # Production dependencies
└── README.md                         # Project documentation
```

---

## Clinical Datasets & Evaluation Scope

RetinaGuard is evaluated across real-world clinical fundus cohorts with clear role division between lesion supervision and domain generalization:

### 1. Active & Evaluated Cohorts (Present Locally in `data/raw/`)

| Dataset | Modality | Samples | Annotations | Evaluation Role in Paper | Status |
|:---|:---|:---:|:---|:---|:---:|
| **IDRiD** | Digital Fundus (Kowa VX-10α) | 516 | **444 Clinical Masks** (MA, HE, EX, SE) + 5-Class Grades | **Primary Benchmark & Lesion Supervision**: Used for spatial lesion attention training, cross-validation, and all attribution grounding metrics (IoU, Dice, Pointing Game Hit Rate). | **Active on Disk & Evaluated** |
| **APTOS 2019** | Multi-Center Photography | 3,662 (1,508 labeled) | Image-level 5-class DR grades | **Cross-Domain Generalization**: Used as the out-of-distribution transfer cohort to evaluate leave-one-center-out resilience without fine-tuning. *(Note: APTOS has no lesion masks).* | **Active on Disk & Evaluated** |

> [!NOTE]
> **Why Attribution IoU is evaluated only on IDRiD**: Measuring attribution alignment (IoU/Dice against clinical truth) requires ground-truth pixel lesion masks. Only IDRiD provides expert-annotated masks for microaneurysms, hemorrhages, and exudates. APTOS only provides global image grades (0–4).

### 2. Extensible Dataset Registry (Planned Large-Scale Cohorts)

The repository provides ingestion scripts and configuration schemas in [`configs/datasets.yaml`](file:///D:/Mini-project/RetinaGuard/configs/datasets.yaml) for researchers with external high-capacity storage (~100+ GB) to plug in additional cohorts:
- **DDR** (13,673 images, multi-hospital lesion masks)
- **EyePACS** (88,702 images, large-scale screening)
- **Messidor-2** (1,748 images, adjudicated DR + DME)

### Standard 5-Class DR Grading Scale (ICDR)
- **Grade 0**: No Diabetic Retinopathy
- **Grade 1**: Mild Non-Proliferative DR (Microaneurysms only)
- **Grade 2**: Moderate NPDR (More than microaneurysms, less than severe)
- **Grade 3**: Severe NPDR (4-2-1 rule: hemorrhages in 4 quadrants, venous beading in 2, or IRMA in 1)
- **Grade 4**: Proliferative DR (Neovascularization, vitreous/preretinal hemorrhage)

---

## Getting Started

### 1. Installation

Clone the repository and install dependencies in a clean Python 3.10+ virtual environment:

```bash
git clone https://github.com/Harsha-Vardan/RetinaGuard.git
cd RetinaGuard

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Dataset Setup & Manifest Generation

Rebuild unified metadata and generate stratified cross-validation manifests:

```bash
# 1. Unify all raw datasets into data/metadata/master_metadata.csv
python -m src.data.build_metadata

# 2. Partition into patient-stratified train, validation, and test sets
python -m src.data.split_data

# 3. Optional: Run comprehensive quality audit
python -m src.data.audit_dataset --dataset IDRiD
```

---

## CLI Reference

### Training the Unified RetinaGuard Framework

Run multi-task training with lesion attention and causal disentanglement:

```bash
# Fast smoke-test verification (2 epochs, CPU-friendly)
python -m src.training.train_retinaguard --smoke-test

# Full multi-task training run
python -m src.training.train_retinaguard \
  --epochs 30 \
  --batch-size 16 \
  --lr 1e-4 \
  --lambda-lesion 0.5 \
  --lambda-adv 0.1 \
  --lambda-cov 0.05
```

### Training Baseline Model

```bash
# Train ResNet18 baseline classifier
python -m src.training.train_baseline --config configs/baseline.yaml
```

### Running Inference

Predict DR grade on any retinal fundus image:

```bash
python -m src.inference.predict \
  --image-path "data/raw/IDRiD/images/IDRiD_001.jpg" \
  --checkpoint "experiments/retinaguard/best_model.pth"
```

---

## Cross-Domain Evaluation

RetinaGuard evaluates models using rigorous clinical metrics:

- **Quadratic Weighted Kappa (QWK)**: Measures ordinal penalty for grade misclassifications.
- **Sensitivity @ 90% Specificity**: High-specificity clinical referability threshold (Grade $\ge 2$).
- **Macro-AUC & Multi-Class AUC**: Area under the ROC curve across all 5 disease severity stages.
- **Expected Calibration Error (ECE)**: Diagnoses model overconfidence under domain shift.
- **Lesion Attribution Intersection-over-Union (Attribution IoU)**: Quantifies alignment between model saliency maps and medical ground-truth lesion masks.

Run cross-domain evaluation:

```bash
python -m src.evaluation.evaluate_cross_domain --config configs/evaluation.yaml
```

---

## Verification & Test Suite

The test suite validates components across numerical correctness, gradient flow, and shape preservation:

```bash
pytest tests -v
```

### Test Coverage (110 / 110 Tests Passing — 100% Pass Rate)

#### A. Scientific & Clinical Research Test Suites
- `tests/test_calibration_uncertainty.py`: Expected Calibration Error (ECE), Multi-Class Brier Score, and normalized entropy metrics.
- `tests/test_robustness_perturbations.py`: Optical blur, illumination shifts, camera sensor noise, and simulated cataract media opacity stress testing.
- `tests/test_statistical_significance.py`: Paired Wilcoxon signed-rank tests, non-parametric permutation tests, and bootstrap 95% CI coverage.
- `tests/test_attribution_sanity.py`: Adebayo et al. parameter randomization sanity checks, Pointing Game boundary tests, and IoU monotonicity.
- `tests/test_causal_invariance.py`: GRL exact gradient inversion check, cross-covariance orthogonality, and multi-task gradient propagation.

#### B. Algorithmic, Pipeline & Architecture Test Suites
- `tests/test_metrics.py`: Quadratic Weighted Kappa, Macro AUC, ECE, Referable Sensitivity @ 90% Spec, and 95% Bootstrap Confidence Intervals.
- `tests/test_models.py`: Baseline backbones (ResNet18, feature extraction, layer freezing).
- `tests/test_advanced_models.py`: ConvNeXt, EfficientNet-B0, and deep architectural adapters.
- `tests/test_preprocessing.py`: FOV segmentation, circular cropping, CLAHE normalization, and blur detection.
- `tests/test_research_models.py`:
  - `SpatialLesionAttention` forward pass & attention map bounds.
  - `LesionGuidedClassifier` masked feature pooling.
  - Auxiliary lesion binary cross-entropy computation.
  - `GradientReversalLayer` backward gradient negation.
  - Cross-covariance statistical independence penalty.
  - `CausalDisentangler` feature projection and dual classification heads.
  - `Unified RetinaGuard` end-to-end multi-task forward & backward pass.
- `tests/test_explainability.py`: Grad-CAM heatmap extraction, Lesion Attention attribution maps, Pointing Game Hit Rate, and IoU/Dice evaluation.
- `tests/test_cross_domain_eval.py`: Leave-One-Center-Out generalization evaluator for Baseline and RetinaGuard with 95% bootstrap CIs.
- `tests/test_ablation_runner.py`: 4-way ablation pipeline across task, lesion, adversarial, and covariance loss variants.
- `tests/test_data_pipeline.py`, `tests/test_dataset_builders.py`, `tests/test_training_datasets.py`: Multi-center dataset loading, balanced sampling, and transforms.
- `tests/test_split.py`: Patient leakage prevention and stratified distribution preservation.
- `tests/test_checkpointing_and_utils.py`: Atomic model save/restore and logging utilities.
- `tests/test_inference_pipeline.py`: Single-image and batch prediction CLI engines.
- `tests/test_ui_server.py`: FastAPI clinical triage endpoints, heatmaps, and report endpoints.

---

## License

This project is licensed under the MIT License for academic and research purposes. Individual medical imaging datasets (IDRiD, APTOS, DDR, EyePACS, Messidor-2) are subject to their respective institutional licenses and data use agreements.

---

## Acknowledgments & Citations

If you utilize RetinaGuard in your research, please cite:

```bibtex
@article{retinaguard2026,
  title   = {RetinaGuard: Lesion-Guided, Causally Informed Diabetic Retinopathy Diagnosis},
  author  = {Busam, Harsha Vardan},
  journal = {arXiv preprint},
  year    = {2026}
}
```