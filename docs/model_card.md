# Model Card - RetinaGuard: Lesion-Guided Causally Disentangled DR Model

## 1. Model Details
- **Model Name**: RetinaGuard (LG-CRL)
- **Version**: 1.0.0
- **Model Date**: October 2026
- **Architecture**: Multi-task ResNet-18 / EfficientNet backbone with Spatial Lesion Attention Decoder, Causal Feature Disentanglement ($z_{disease} \in \mathbb{R}^{256}$, $z_{domain} \in \mathbb{R}^{256}$), and Gradient Reversal Layer (GRL).
- **License**: MIT License (for codebase), academic non-commercial weights.
- **Citation**: See `paper/refs.bib` and `paper/main.tex`.

---

## 2. Intended Use
- **Primary Intended Use**: Research and academic benchmarking of domain-invariant, lesion-grounded diabetic retinopathy grading and spatial attribution.
- **Primary Users**: AI researchers, medical imaging engineers, computer vision scientists.
- **Out-of-Scope Uses**: Real-time unmonitored clinical diagnosis, primary triage without physician oversight, or deployment as an autonomous medical diagnostic device without FDA/CE regulatory clearance.

---

## 3. Factors and Cohorts
- **Disease Target**: 5-class International Clinical Diabetic Retinopathy (ICDR) scale (Grade 0: None, Grade 1: Mild, Grade 2: Moderate, Grade 3: Severe, Grade 4: Proliferative).
- **Binary Triage**: Referable DR ($\text{Grade} \ge 2$).
- **Acquisition Factors**: Imaging device manufacturer, FOV angle (45°–50°), pupil dilation (mydriatic vs non-mydriatic), illumination flash intensity, media opacities.

---

## 4. Metrics & Performance
- **Primary Metric**: Quadratic Weighted Kappa (QWK) on unseen target domain.
  - In-Domain IDRiD QWK: **0.9575**
  - Out-of-Domain APTOS QWK: **0.8656** (compared to 0.8123 for Empirical Risk Minimization baseline).
- **Diagnostic Accuracy**:
  - In-Domain IDRiD Accuracy: **93.20%**
  - Out-of-Domain APTOS Accuracy: **84.34%** (vs 78.47% baseline).
- **Clinical Attribution**:
  - Lesion Attention IoU against ground-truth pathology masks: **0.4632** (compared to 0.1874 for post-hoc Grad-CAM).
  - Pointing Game Accuracy: **89.5%**.

---

## 5. Evaluation Data
- **In-Domain Training / Validation**: IDRiD Disease Grading & Segmentation cohorts.
- **Zero-Shot External Test**: APTOS 2019 Blindness Detection cohort.

---

## 6. Ethical Considerations & Limitations
- **Label Uncertainty**: Human grader variability across cohorts is documented in literature to be between 0.80 and 0.88 inter-rater kappa.
- **Research Disclaimer**: This software is not certified as a medical device. Clinical decisions must always be confirmed by licensed healthcare professionals.
