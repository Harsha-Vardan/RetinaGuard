# Dataset Audit Report

Generated: 2026-10-13 22:09:05

## APTOS
- Path: `D:\Mini-project\data\raw\APTOS`
- Total images: 25
- Valid images: 25
- Corrupted: 0
- Image dimensions: {'512x512x3': 25}
- Label files found: True
- Mask files: 0
- Exact duplicates: 0
- Quality flags: blur=0, dark=0, bright=0

### Grade Distribution

**train.csv**:
| Grade | Count |
|-------|-------|
| 0 | 4 |
| 1 | 4 |
| 2 | 3 |
| 3 | 3 |
| 4 | 3 |

## IDRiD
- Path: `D:\Mini-project\data\raw\IDRiD`
- Total images: 1032
- Valid images: 1032
- Corrupted: 0
- Image dimensions: {'4288x2848x3': 1032}
- Label files found: True
- Mask files: 444
- Exact duplicates: 522
- Quality flags: blur=1032, dark=6, bright=0

### Grade Distribution

**a. IDRiD_Disease Grading_Training Labels.csv**:
| Grade | Count |
|-------|-------|
| 0 | 134 |
| 1 | 20 |
| 2 | 136 |
| 3 | 74 |
| 4 | 49 |

**b. IDRiD_Disease Grading_Testing Labels.csv**:
| Grade | Count |
|-------|-------|
| 0 | 34 |
| 1 | 5 |
| 2 | 32 |
| 3 | 19 |
| 4 | 13 |

### Annotation Coverage
- Matched image-mask pairs: 0
- Unmatched masks: 444
- Images without masks: 413

### Issues
- ⚠️ 522 exact duplicate images
