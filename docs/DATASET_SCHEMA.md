# Dataset Schema Documentation

Generated: 2026-10-15 22:49:29

Total records: 3530

## Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| image_id | str | Stable unique identifier (dataset_stem) |
| image_path | str | Path to original image |
| dataset_id | str | Source dataset name |
| dr_grade | int | Standardized DR grade 0-4 |
| original_label | int | Label as provided by source |
| hospital_id | str | Hospital identifier (if available) |
| camera_id | str | Imaging device (if available) |
| patient_id | str | Patient identifier (if available) |
| eye | str | Left/right eye (if available) |
| image_quality | float | Quality score (if computed) |
| gradable | bool | Whether image is gradable |
| lesion_mask_path | str | Path to lesion mask (if available) |
| lesion_annotation_type | str | Type of annotation |
| split | str | train/val/test/excluded |
| duplicate_group | str | Duplicate group ID |
| label_status | str | verified/missing/conflicting/uncertain |
| preprocessing_version | str | Applied processing version |

## DR Grade Mapping

| Grade | Description |
|-------|-------------|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |

## Per-Dataset Summary

### APTOS
- Records: 2714
- With labels: 1508
- With masks: 0
- Grade distribution:
  - Grade 0: 308
  - Grade 1: 300
  - Grade 2: 300
  - Grade 3: 300
  - Grade 4: 300

### DDR
- Records: 100
- With labels: 100
- With masks: 0
- Grade distribution:
  - Grade 0: 20
  - Grade 1: 20
  - Grade 2: 20
  - Grade 3: 20
  - Grade 4: 20

### EyePACS
- Records: 100
- With labels: 100
- With masks: 0
- Grade distribution:
  - Grade 0: 23
  - Grade 1: 21
  - Grade 2: 19
  - Grade 3: 20
  - Grade 4: 17

### IDRiD
- Records: 516
- With labels: 516
- With masks: 81
- Grade distribution:
  - Grade 0: 168
  - Grade 1: 25
  - Grade 2: 168
  - Grade 3: 93
  - Grade 4: 62

### Messidor-2
- Records: 100
- With labels: 80
- With masks: 0
- Grade distribution:
  - Grade 1: 20
  - Grade 2: 20
  - Grade 3: 20
  - Grade 4: 20
