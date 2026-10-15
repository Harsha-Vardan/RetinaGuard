# Dataset Acquisition Report

Generated: 2026-10-15 22:32:45

## IDRiD ✅
- **Status**: `VERIFIED`
- **Official URL**: https://idrid.grand-challenge.org/
- **License**: CC BY 4.0
- **Local path**: `D:\Mini-project\RetinaGuard\data\raw\IDRiD`
- **Images found**: 1032
- **Labels found**: True
- **Annotations found**: True
- **Masks found**: 444
- **Subdirectories**:
  - `a. Training Set`: 826 images
  - `b. Testing Set`: 206 images

## APTOS ✅
- **Status**: `VERIFIED`
- **Official URL**: https://www.kaggle.com/competitions/aptos2019-blindness-detection
- **License**: Kaggle Competition Rules
- **Local path**: `D:\Mini-project\RetinaGuard\data\raw\APTOS`
- **Images found**: 1514
- **Labels found**: True
- **Annotations found**: True
- **Masks found**: 1200
- **Subdirectories**:
  - `test_images`: 8 images
  - `train_images`: 1506 images

## DDR 📋
- **Status**: `REQUIRES_MANUAL_APPROVAL`
- **Official URL**: https://github.com/nkicsl/DDR-dataset
- **License**: Research use
- **Local path**: `None`
- **Images found**: 0
- **Labels found**: False
- **Annotations found**: False
- **Required actions**:
  - Dataset DDR requires manual download:
  -   Official source: https://github.com/nkicsl/DDR-dataset
  -   Place files in: D:\Mini-project\RetinaGuard\data\raw\DDR
  -   Re-run download_datasets after placing the files.

## EyePACS 🔐
- **Status**: `REQUIRES_AUTHENTICATION`
- **Official URL**: https://www.kaggle.com/competitions/diabetic-retinopathy-detection
- **License**: Kaggle Competition Rules
- **Local path**: `None`
- **Images found**: 0
- **Labels found**: False
- **Annotations found**: False
- **Required actions**:
  - Create Kaggle API token: https://www.kaggle.com/settings -> API -> Create New Token
  - Save kaggle.json to C:\Users\User\.kaggle\kaggle.json
  - OR set KAGGLE_USERNAME and KAGGLE_KEY environment variables
  - Accept competition rules at: https://www.kaggle.com/competitions/diabetic-retinopathy-detection
