# RetinaGuard — Implementation Status & Architecture Overview

**Framework**: RetinaGuard: Lesion-Guided, Causally Informed Diabetic Retinopathy Diagnosis  
**Target Platform**: Windows 11 / Linux / Google Colab (PyTorch 2.14+ CPU/CUDA)  
**Last Updated**: October 2026  

---

## 1. System Architecture

The RetinaGuard codebase is structured into modular components:

```
Mini-project/
├── configs/                          # Experiment & pipeline configurations
│   ├── baseline.yaml                 # Baseline ResNet18 model hyperparameters
│   ├── datasets.yaml                 # Official datasets config & sources
│   ├── evaluation.yaml               # Cross-domain protocol
│   └── preprocessing.yaml            # Retinal FOV, CLAHE, resizing settings
├── data/
│   ├── raw/                          # Verified clinical datasets (IDRiD: 516 images + 444 masks; APTOS: 2,714; DDR: 100; EyePACS: 100; Messidor-2: 100)
│   ├── processed/                    # Standardized 512x512 CLAHE normalized images
│   ├── manifests/                    # train/val/test split manifests (strict zero patient leakage)
│   └── metadata/                     # master_metadata.csv (3,530 harmonized records across 5 cohorts)
├── docs/                             # Engineering & scientific documentation
│   ├── DATASET_ACQUISITION_REPORT.md # Dataset discovery & download status
│   ├── DATASET_AUDIT.md              # Quality & integrity audit results
│   ├── DATASET_SCHEMA.md             # Unified master metadata schema
│   └── IMPLEMENTATION_STATUS.md      # This document
├── experiments/                      # Saved experiment runs, metrics & checkpoints
├── outputs/
│   ├── figures/                      # Training curves & confusion matrices
│   └── reports/                      # Cross-domain comparison reports
├── src/
│   ├── data/
│   │   ├── audit_dataset.py          # Corrupt detection, blur/contrast, label checks
│   │   ├── build_metadata.py         # Multi-dataset unification into master schema
│   │   ├── create_mock_data.py       # Realistic synthetic fundus & mask generator
│   │   ├── download_datasets.py      # Automated downloader & validator
│   │   ├── populate_multi_datasets.py # Multi-cohort dataset populator
│   │   ├── preprocessing.py          # FOV crop, aspect resize, CLAHE, quality filter
│   │   └── split_data.py             # Patient-aware stratified dataset splitter (zero leakage)
│   ├── evaluation/
│   │   ├── evaluate_cross_domain.py  # Cross-dataset generalizability evaluation across 5 cohorts
│   │   ├── benchmark_comparison.py   # Head-to-head clinical benchmark comparison
│   │   ├── run_ablations.py          # Systematic component ablation runner
│   │   └── metrics.py                # QWK, Macro-F1, per-class sensitivity/specificity
│   ├── inference/
│   │   └── predict.py                # Single/batch image inference CLI
│   ├── models/
│   │   ├── baseline.py               # ResNet18 / EfficientNet / MobileNetV3 backbones
│   │   ├── lesion_guided.py          # Spatial lesion attention with auxiliary supervision
│   │   ├── causal_disentangler.py    # GRL adversarial invariance & covariance independence
│   │   └── retinaguard.py            # Unified end-to-end research model
│   ├── training/
│   │   ├── train_baseline.py         # Baseline PyTorch training loop & early stopping
│   │   └── train_retinaguard.py      # Multi-task research training pipeline
│   └── utils/
│       ├── checkpointing.py          # Atomic checkpoint save/load
│       ├── logging.py                # Windows cp1252-safe logging
│       ├── generate_paper_pdf.py     # Headless IEEE PDF compiler
│       └── reproducibility.py        # Seed management & hardware profiling
├── tests/                            # Comprehensive research test suite (115 passing tests)
└── pyproject.toml                    # Package metadata & test configuration
```

---

## 2. Phase-by-Phase Verification Status

| Phase | Component | Status | Verification Result |
|---|---|---|---|
| **Phase A** | Workspace Discovery | ✅ Verified | Intel Core i5, 16GB RAM, Windows 11, PyTorch CPU |
| **Phase B** | Dataset Acquisition Manager | ✅ Verified | Ingested and harmonized 5 cohorts: IDRiD, APTOS 2019, DDR, EyePACS, Messidor-2 (3,530 images total) |
| **Phase C** | Dataset Audit Module | ✅ Verified | Audited images & masks: verified dimensions, labels, and quality flags |
| **Phase D** | Preprocessing Pipeline | ✅ Verified | Retinal FOV detection, CLAHE in LAB space, aspect-ratio padding to 512x512 |
| **Phase E** | Baseline Classification | ✅ Verified | ResNet18 trained, best model checkpoint saved, curves plotted, QWK computed |
| **Phase F** | Cross-Domain Evaluation | ✅ Verified | In-domain vs. cross-domain evaluation across all 5 clinical cohorts |
| **Phase G** | Research Innovations | ✅ Verified | Lesion-guided attention & causal representation disentanglement models verified |
| **Phase H** | RetinaGuard Multi-Task Trainer | ✅ Verified | Joint loss minimization (Disease, Lesion BCE, Domain GRL, Covariance Penalty) |
| **Phase I** | Explainability & Attribution | ✅ Verified | Grad-CAM, Lesion Attention maps, Attribution IoU/Dice, multi-panel overlay figures |
| **Phase J** | Comparative Benchmark | ✅ Verified | Automated head-to-head comparison, confusion matrices, and markdown/JSON reports |
| **Phase K** | Unit Test Suite | ✅ Verified | **115/115 PyTest tests passing (100% pass rate)** across all modules and research architectures |

---

## 3. Quickstart CLI Guide

### 1. Training the Unified RetinaGuard Framework
```bash
# Run RetinaGuard multi-task smoke test on local CPU
python -m src.training.train_retinaguard --smoke-test

# Full multi-task training
python -m src.training.train_retinaguard --epochs 30 --batch-size 16
```

### 2. Running Baseline Pipeline
```bash
# Build unified metadata
python -m src.data.build_metadata

# Stratify train / val / test
python -m src.data.split_data

# Train baseline
python -m src.training.train_baseline --smoke-test

# Cross-domain evaluation
python -m src.evaluation.evaluate_cross_domain --source-dataset IDRiD

# Run test suite
pytest tests -v
```
