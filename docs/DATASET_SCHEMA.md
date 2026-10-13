# Dataset Schema Documentation

Generated: 2026-10-13 21:35:55

Total records: 50

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
- Records: 25
- With labels: 17
- With masks: 0
- Grade distribution:
  - Grade 0: 4
  - Grade 1: 4
  - Grade 2: 3
  - Grade 3: 3
  - Grade 4: 3

### IDRiD
- Records: 25
- With labels: 25
- With masks: 0
- Grade distribution:
  - Grade 0: 5
  - Grade 1: 5
  - Grade 2: 5
  - Grade 3: 5
  - Grade 4: 5
