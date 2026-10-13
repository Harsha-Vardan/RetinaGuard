# RetinaGuard — Project Roadmap & Tomorrow's Execution Plan

**Project Name**: RetinaGuard: Lesion-Guided, Causally Informed Diabetic Retinopathy Diagnosis  
**Repository**: [https://github.com/Harsha-Vardan/RetinaGuard](https://github.com/Harsha-Vardan/RetinaGuard)  
**Status**: Core Framework Complete (Phases A–K Verified, 27/27 Tests Passing)  
**Execution Target**: Complete remaining research & deployment deliverables  
**Created**: October 2026  

---

## 1. Current State Summary

The foundational engineering and research framework is fully implemented, verified, and committed:

- **Data Pipeline**: Real IDRiD dataset ingested (516 images, 444 lesion masks) with circular FOV cropping, aspect-ratio padding, and LAB CLAHE enhancement.
- **Model Architectures**: 
  - Standard ResNet18 / EfficientNet baselines.
  - Spatial Lesion Attention module with auxiliary BCE segmentation supervision.
  - Causal Disentanglement with Gradient Reversal Layer (GRL) and covariance independence penalty ($Z_{dis} \perp Z_{dom}$).
  - Unified end-to-end RetinaGuard model.
- **Explainability**: Grad-CAM, spatial attention extraction, and attribution IoU/Dice scoring against clinical lesion masks.
- **Benchmarking & Inference**: Cross-domain evaluation suite, comparative benchmarking reports, and CLI predictor (`predict.py`).
- **Test Suite**: **27 / 27 unit tests passing** across all modules.

---

## 2. Tomorrow's Execution Plan & Deliverables

To take the project from research codebase to a complete, publication-ready and deployable system, we will execute four remaining modules tomorrow:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   TOMORROW'S EXECUTION ROADMAP                         │
├────────────────────────────────┬───────────────────────────────────────┤
│ 1. Interactive Clinical Web UI │ Streamlit / Gradio Doctor's Dashboard │
│ 2. Google Colab GPU Notebook   │ 1-Click End-to-End Training (CUDA)    │
│ 3. Systematic Ablation Suite   │ 4-Way Quantitative Component Study    │
│ 4. IEEE Research Paper Draft   │ Publication-Ready Academic Manuscript │
└────────────────────────────────┴───────────────────────────────────────┘
```

---

### Module 1: Interactive Clinical Web UI & Diagnostic Dashboard

**Goal**: Build a browser-based clinical interface enabling clinicians and evaluators to interact with RetinaGuard in real time without command-line usage.

- [ ] **1.1 UI Framework Setup**: Implement a responsive web application (`src/ui/app.py`) using **Streamlit** (or Gradio).
- [ ] **1.2 Image Ingestion & Preprocessing**:
  - Drag-and-drop fundus photograph uploader (`.jpg`, `.png`, `.tif`).
  - Real-time side-by-side display of Raw Fundus vs. Preprocessed FOV-cropped CLAHE image.
- [ ] **1.3 Model Inference Selector**:
  - Toggle between **Standard Baseline (ResNet18)** and **RetinaGuard (Lesion-Guided & Disentangled)**.
- [ ] **1.4 Diagnostic Summary Card**:
  - Predicted DR Grade (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative) with ICDR clinical description.
  - **Referable DR Risk Alert**: Visual banner (Green: Non-Referable, Red: Referable DR Triage Recommendation).
  - Class confidence probability bar chart.
- [ ] **1.5 Explainable AI (XAI) Panel**:
  - Interactive multi-panel visualization:
    1. Input Fundus Image.
    2. Model Grad-CAM activation heatmap.
    3. Lesion-guided attention map highlighting microaneurysms, hemorrhages, and exudates.
    4. Clinical attribution overlay.
- [ ] **1.6 Report Export**:
  - One-click generation of a downloadable clinical summary report (PDF / PNG).

---

### Module 2: Google Colab 1-Click GPU Training Notebook

**Goal**: Provide a zero-friction Jupyter Notebook (`notebooks/RetinaGuard_Colab_Training.ipynb`) allowing anyone to train RetinaGuard on free Google Colab GPUs (T4 / V100).

- [ ] **2.1 Environment Setup Cell**:
  - Automated GPU detection (`torch.cuda.is_available()`), PyTorch version logging, dependency installation.
- [ ] **2.2 Dataset Download & Extraction**:
  - Scripted download of IDRiD and Kaggle APTOS datasets into Colab `/content/data/`.
  - Automated metadata generation and patient-aware stratification.
- [ ] **2.3 Mixed-Precision (AMP) Training**:
  - PyTorch `torch.cuda.amp.autocast()` training loops for 2–3x faster GPU training.
- [ ] **2.4 Checkpoint & Metric Tracking**:
  - Real-time loss and QWK metric curve plotting.
  - Automatic saving of best model weights to Google Drive.
- [ ] **2.5 Automated Evaluation & Visualization Cell**:
  - End-of-training benchmark comparison generating confusion matrices and attribution samples.

---

### Module 3: Systematic Ablation Experiment Suite

**Goal**: Scientifically validate each individual contribution of RetinaGuard by running a systematic ablation study and producing publication tables.

- [ ] **3.1 Script Implementation (`src/evaluation/run_ablations.py`)**:
  - Automated runner executing the 4 configurations on identical data splits:
    1. **Baseline ResNet18**: Standard cross-entropy classification ($\mathcal{L}_{task}$).
    2. **ResNet18 + Lesion Attention**: Supervised spatial attention ($\mathcal{L}_{task} + \lambda_1 \mathcal{L}_{lesion}$).
    3. **ResNet18 + Causal Disentanglement**: Adversarial domain invariance & covariance independence ($\mathcal{L}_{task} + \lambda_2 \mathcal{L}_{adv} + \lambda_3 \mathcal{L}_{cov}$).
    4. **Unified RetinaGuard**: Complete proposed framework ($\mathcal{L}_{total}$).
- [ ] **3.2 Quantitative Comparison Metrics**:
  - Quadratic Weighted Kappa (QWK).
  - Macro-F1 score.
  - Cross-domain accuracy drop ($\Delta = Acc_{in-domain} - Acc_{out-of-domain}$).
  - Clinical Attribution IoU and Dice score against true lesion masks.
- [ ] **3.3 Automated Table Generation**:
  - Export structured outputs to `outputs/reports/ablation_study.csv` and formatted markdown for reports.

---

### Module 4: Academic Project Report / IEEE Conference Paper Draft

**Goal**: Document the methodology, causal formulations, and experimental findings in a structured IEEE-style research paper format (`docs/RetinaGuard_IEEE_Paper.md` / PDF).

- [ ] **4.1 Abstract & Introduction**:
  - Problem definition: Shortcut learning and domain shifts in deep learning DR diagnosis.
  - Research objective and key contributions.
- [ ] **4.2 Structural Causal Model (SCM)**:
  - Directed Acyclic Graph (DAG) formulation:
    - Input $X \rightarrow \{Z_{dis}, Z_{dom}\}$.
    - Disease grade $Y \leftarrow Z_{dis}$.
    - Domain/Hospital index $D \leftarrow Z_{dom}$.
    - Statistical independence constraint: $Z_{dis} \perp Z_{dom}$.
- [ ] **4.3 Mathematical Methodology & Objective Function**:
  - Formulation of joint loss:
    $$\mathcal{L}_{total} = \mathcal{L}_{CE}(y, \hat{y}) + \lambda_1 \mathcal{L}_{BCE}(M, \hat{M}) + \lambda_2 \mathcal{L}_{dom}(d, \hat{d}) + \lambda_3 \mathcal{L}_{cov}(Z_{dis}, Z_{dom})$$
- [ ] **4.4 Experimental Results & Benchmarking**:
  - In-domain vs. cross-domain performance tables.
  - Attribution localization accuracy (IoU / Dice).
  - Ablation comparison analysis.
- [ ] **4.5 Clinical Relevance & Discussion**:
  - Trustworthiness, false negative reduction in referable DR, and translational clinical value.

---

## 3. Tomorrow's Step-by-Step Execution Sequence

| Order | Phase | Estimated Scope | Output Files |
|:---:|---|---|---|
| **Step 1** | **Interactive Web UI** | UI application & clinical widgets | `src/ui/app.py`, demo screenshot |
| **Step 2** | **Google Colab Notebook** | GPU training notebook with documentation | `notebooks/RetinaGuard_Colab_Training.ipynb` |
| **Step 3** | **Ablation Experiment Runner** | Automated 4-way evaluation script | `src/evaluation/run_ablations.py`, `outputs/reports/ablation_study.csv` |
| **Step 4** | **IEEE Research Paper Draft** | Full paper manuscript & tables | `docs/RetinaGuard_IEEE_Paper.md` |
| **Step 5** | **Final Git Sync & Project Review** | Git commit, push, and validation | Clean repo state, synced with GitHub |

---

## 4. Acceptance Criteria for 100% Completion

The project will be considered 100% finished when:
1. `streamlit run src/ui/app.py` launches a functional diagnostic dashboard allowing drag-and-drop inference with heatmaps.
2. `notebooks/RetinaGuard_Colab_Training.ipynb` runs end-to-end on Colab with zero errors.
3. `python -m src.evaluation.run_ablations` executes all 4 variants and outputs the ablation comparison table.
4. `docs/RetinaGuard_IEEE_Paper.md` contains a complete academic manuscript ready for submission or presentation.
5. All tests (`pytest tests -v`) remain 100% passing and all changes are committed and pushed to GitHub.
