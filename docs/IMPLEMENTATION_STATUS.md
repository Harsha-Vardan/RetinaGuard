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
│   ├── raw/                          # Raw datasets (IDRiD, APTOS, DDR, EyePACS)
│   ├── processed/                    # Standardized 512x512 CLAHE normalized images
│   ├── manifests/                    # train/val/test split manifests
│   └── metadata/                     # master_metadata.csv & acquisition status
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
│   │   ├── preprocessing.py          # FOV crop, aspect resize, CLAHE, quality filter
│   │   └── split_data.py             # Patient-aware stratified dataset splitter
│   ├── evaluation/
│   │   ├── evaluate_cross_domain.py  # Cross-dataset generalizability evaluation
│   │   └── metrics.py                # QWK, Macro-F1, per-class sensitivity/specificity
│   ├── inference/
│   │   └── predict.py                # Single/batch image inference CLI
│   ├── models/
│   │   └── baseline.py               # ResNet18 / EfficientNet / MobileNetV3 backbones
│   ├── training/
│   │   └── train_baseline.py         # PyTorch training loop, checkpoints, early stopping
│   └── utils/
│       ├── checkpointing.py          # Atomic checkpoint save/load
│       ├── logging.py                # Windows cp1252-safe logging
│       └── reproducibility.py        # Seed management & hardware profiling
├── tests/                            # PyTest unit test suite
└── pyproject.toml                    # Package metadata & test configuration
```

---

## 2. Phase-by-Phase Verification Status

| Phase | Component | Status | Verification Result |
|---|---|---|---|
| **Phase A** | Workspace Discovery | ✅ Verified | Intel Core i5, 16GB RAM, Windows 11, PyTorch CPU |
| **Phase B** | Dataset Acquisition Manager | ✅ Verified | Automated search, checksum, extraction, and instructions for IEEE DataPort & Kaggle |
| **Phase C** | Dataset Audit Module | ✅ Verified | Discovered images, label CSVs, masks, dimensions; 0 corrupt files |
| **Phase D** | Preprocessing Pipeline | ✅ Verified | Retinal FOV detection, CLAHE in LAB space, aspect-ratio padding to 512x512 |
| **Phase E** | Baseline Classification | ✅ Verified | ResNet18 trained, best model checkpoint saved, curves plotted, QWK computed |
| **Phase F** | Cross-Domain Evaluation | ✅ Verified | In-domain vs. cross-domain evaluation, confusion matrix heatmaps |
| **Phase G** | Inference CLI | ✅ Verified | `src.inference.predict` outputs grade, description & softmax distribution |
| **Phase H** | Unit Test Suite | ✅ Verified | 14/14 PyTest tests passing across metrics, models, preprocessing & splits |

---

## 3. Quickstart CLI Guide

### 1. Run Pipeline End-to-End
```bash
# Generate mock dataset for local smoke testing
python -m src.data.create_mock_data --num-idrid 25 --num-aptos 25

# Audit raw datasets
python -m src.data.audit_dataset

# Build unified metadata
python -m src.data.build_metadata

# Create train/val/test splits
python -m src.data.split_data

# Run preprocessing
python -m src.data.preprocessing

# Run baseline training smoke test
python -m src.training.train_baseline --smoke-test

# Run single image prediction
python -m src.inference.predict --image path/to/image.jpg --checkpoint experiments/baseline/<exp_id>/best_model.pth

# Run cross-domain evaluation
python -m src.evaluation.evaluate_cross_domain --source-dataset IDRiD

# Run test suite
pytest tests -v
```

### 2. Acquiring Real Datasets
- **IDRiD**: Download `A. Disease Grading`, `B. Localization`, `C. Segmentation` from [IEEE DataPort](https://idrid.grand-challenge.org/) into `data/raw/IDRiD/`
- **APTOS**: Place `kaggle.json` into `C:\Users\<User>\.kaggle\kaggle.json` or set `KAGGLE_USERNAME` & `KAGGLE_KEY`, then run `python -m src.data.download_datasets --dataset APTOS`
