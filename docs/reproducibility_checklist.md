# CLAIM-Style Reproducibility Checklist for Medical AI

Based on the **Checklist for Artificial Intelligence in Medical Imaging (CLAIM)** guidelines:

| Item # | Guideline Requirement | RetinaGuard Implementation & Verification |
|:---:|:---|:---|
| **1** | Full study design and diagnostic objective stated | Formalized in `PLAN.md` and `paper/main.tex` (5-class ICDR grading + binary referable DR triage). |
| **2** | Data source and eligibility criteria specified | Documented in `docs/DATA_ACCESS.md` and `docs/datasheet.md`. |
| **3** | Ground truth annotation protocol detailed | Expert ophthalmologist grading protocols reported for IDRiD and APTOS cohorts. |
| **4** | Patient-level data split enforcement | Verified in `tests/test_splits_no_leakage.py`; patient hashing prevents intra-patient train/test leakage. |
| **5** | Preprocessing, border cropping, and normalization | Implemented in `src/data/preprocessing.py` (Otsu FOV extraction, Graham local contrast, CLAHE). |
| **6** | Model architecture and loss formulations | Documented in `src/models/retinaguard.py` and `configs/retinaguard.yaml`. |
| **7** | Random seeds and computational environment | Fixed deterministic seeds (seed=42) in all configs and scripts; logged in `experiments/`. |
| **8** | Hyperparameter tuning and model selection criteria | Validation QWK used for model selection; test sets strictly frozen during training. |
| **9** | Statistical significance and confidence intervals | Bootstrap 95% CIs (n=1000 resamples), DeLong test for AUROC, Wilcoxon signed-rank tests implemented in `src/evaluation/calibration.py` and `tests/test_statistical_significance.py`. |
| **10** | Explainability and attribution sanity checks | Evaluated via Adebayo et al. parameter randomization, pointing game, and ground-truth lesion IoU in `tests/test_attribution_sanity.py`. |
| **11** | Failure case analysis and clinical limitations | Documented in `docs/limitations.md`, confusion matrices, and multi-panel diagnostic overlays. |
| **12** | Open-source code availability | Full clean codebase with automated test suites provided under MIT License. |
