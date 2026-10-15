# RetinaGuard - Dataset Access and Acquisition Guide

This document describes the official sources, licensing, registration, and local directory configuration for all datasets incorporated into the RetinaGuard research framework.

---

## 1. Primary Evaluated Clinical Cohorts

### 1.1 IDRiD (Indian Diabetic Retinopathy Image Dataset)
- **Official Source**: IEEE Dataport ([DOI: 10.21227/H29W98](https://doi.org/10.21227/H29W98)) / Grand Challenge.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Access Procedure**: Free registration on IEEE DataPort or Grand Challenge portal.
- **Local Structure**:
  ```text
  data/raw/IDRiD/
  ├── B. Disease Grading/
  │   ├── 1. Original Images/
  │   └── 2. Groundtruths/
  └── A. Segmentation/
      ├── 1. Original Images/
      └── 2. All Segmentation Groundtruths/
          ├── 1. Microaneurysms/
          ├── 2. Haemorrhages/
          ├── 3. Hard Exudates/
          └── 4. Soft Exudates/
  ```
- **Role in Study**: Primary benchmark for pixel-level lesion segmentation, lesion-guided spatial attention, and clinical attribution evaluation.

### 1.2 APTOS 2019 Blindness Detection
- **Official Source**: Kaggle Competition ([APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection))
- **Provider**: Aravind Eye Hospital, India.
- **License**: Competition terms / Non-commercial academic research use.
- **Access Procedure**: Kaggle account registration and acceptance of competition rules; programmatic download via `kaggle competitions download -c aptos2019-blindness-detection`.
- **Local Structure**:
  ```text
  data/raw/APTOS/
  ├── train_images/
  ├── test_images/
  └── train.csv
  ```
- **Role in Study**: Unseen out-of-domain evaluation cohort for cross-hospital external validation.

---

## 2. Multi-Cohort Cross-Domain Evaluation Cohorts

### 2.1 DDR (Dataset for Diabetic Retinopathy)
- **Official Source**: Institute of Software, Chinese Academy of Sciences / [DDR Dataset](https://github.com/nkicsl/DDR-dataset).
- **License**: Research use only upon academic request.
- **Local Structure**:
  ```text
  data/raw/DDR/
  ├── train/
  ├── test/
  ├── train.csv
  └── lesion_segmentation/
  ```
- **Role in Study**: Cross-domain generalization benchmark with paired multi-lesion segmentation masks for cross-hospital lesion transfer evaluation.

### 2.2 EyePACS (Kaggle Diabetic Retinopathy 2015)
- **Official Source**: [EyePACS / Kaggle DR Competition](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- **License**: Research use only.
- **Local Structure**:
  ```text
  data/raw/EyePACS/
  ├── images/
  └── trainLabels.csv
  ```
- **Role in Study**: Primary care screening tele-ophthalmology cohort with paired bilateral eyes (left/right) per patient, used to benchmark zero patient-level split leakage.

### 2.3 Messidor-2
- **Official Source**: ADCIS / Messidor-2 Consortium ([Messidor-2](https://www.adcis.net/en/third-party/messidor2/))
- **License**: Research agreement / academic research use.
- **Local Structure**:
  ```text
  data/raw/Messidor-2/
  ├── images/
  └── messidor_data.csv
  ```
- **Role in Study**: European multi-center clinical cohort (Brest, Paris, Saint-Étienne) captured via Topcon TRC-NW6 cameras with expert adjudicated DR and DME grades.

---

## 3. Private Multi-Hospital Clinical Data Protocol

- **Ethics / IRB**: Multi-site clinical studies require IRB approval from the respective institutional ethics review board.
- **De-Identification**: All DICOM and EXIF metadata must be stripped prior to ingestion using `src/data/dedup.py`. Patient IDs are hashed using salted SHA-256 (`hash_patient_id`).
- **Storage**: Never commit raw patient fundus images to public version control or cloud repositories without strict encryption and compliance approvals.
