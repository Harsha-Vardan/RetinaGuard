"""RetinaGuard - Unified Metadata Builder.

Builds master_metadata.csv from all available datasets, unifying labels
and paths into a single consistent schema.

Usage:
    python -m src.data.build_metadata
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger("build_metadata")

# Standard 5-class DR grade mapping
DR_GRADE_MAP = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}


class MetadataBuilder:
    """Builds unified metadata from all available datasets."""

    def __init__(self, data_root: str | Path | None = None):
        self.data_root = Path(data_root or PROJECT_ROOT / "data")
        self.raw_dir = self.data_root / "raw"
        self.metadata_dir = self.data_root / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def build(self) -> pd.DataFrame:
        """Build the master metadata table from all available datasets.

        Returns:
            DataFrame with unified metadata.
        """
        all_records = []

        # Process each dataset
        for dataset_dir in sorted(self.raw_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            name = dataset_dir.name
            logger.info(f"Processing dataset: {name}")

            if name == "IDRiD":
                records = self._process_idrid(dataset_dir)
            elif name == "APTOS":
                records = self._process_aptos(dataset_dir)
            elif name == "DDR":
                records = self._process_ddr(dataset_dir)
            elif name == "EyePACS":
                records = self._process_eyepacs(dataset_dir)
            else:
                records = self._process_generic(name, dataset_dir)

            all_records.extend(records)
            logger.info(f"  -> {len(records)} records from {name}")

        if not all_records:
            logger.warning("No records found. Ensure datasets are downloaded and extracted.")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(all_records)

        # Ensure required columns exist
        required_cols = [
            "image_id", "image_path", "dataset_id", "dr_grade", "original_label",
            "hospital_id", "camera_id", "patient_id", "eye",
            "image_quality", "gradable", "lesion_mask_path",
            "lesion_annotation_type", "split", "duplicate_group",
            "label_status", "preprocessing_version",
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        # Reorder columns
        df = df[[c for c in required_cols if c in df.columns] +
                [c for c in df.columns if c not in required_cols]]

        # Save
        output_path = self.metadata_dir / "master_metadata.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved master metadata: {output_path} ({len(df)} records)")

        # Save schema documentation
        self._save_schema_docs(df)

        return df

    def _process_idrid(self, dataset_dir: Path) -> list[dict]:
        """Process IDRiD dataset into metadata records."""
        import re
        records = []

        # 1. Locate grading directory
        grading_dir = dataset_dir / "B. Disease Grading"
        if not grading_dir.exists():
            for p in dataset_dir.iterdir():
                if p.is_dir() and "grading" in p.name.lower():
                    grading_dir = p
                    break
            else:
                grading_dir = dataset_dir

        seg_dir = dataset_dir / "A. Segmentation"
        if not seg_dir.exists():
            for p in dataset_dir.iterdir():
                if p.is_dir() and "segmentation" in p.name.lower():
                    seg_dir = p
                    break
            else:
                seg_dir = dataset_dir

        # 2. Index lesion segmentation masks from segmentation directory
        masks_map: dict[tuple[str, int], list[Path]] = {}
        mask_stem_index: dict[str, Path] = {}
        if seg_dir.exists():
            for m in seg_dir.rglob("*.tif"):
                split = "train" if "training" in str(m).lower() else "test"
                num_match = re.search(r"\d+", m.stem)
                if num_match:
                    num = int(num_match.group())
                    masks_map.setdefault((split, num), []).append(m)
                mask_stem_index[m.stem] = m

        # 3. Read grading labels separated by split
        grading_labels: dict[tuple[str, str], int] = {}
        for lf in dataset_dir.rglob("*.csv"):
            if "grading" not in lf.name.lower() and "disease" not in lf.name.lower() and "label" not in lf.name.lower():
                continue
            split = "train" if "train" in lf.name.lower() else ("test" if "test" in lf.name.lower() else "unknown")
            try:
                df = pd.read_csv(lf)
                name_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["image", "name", "id"])]
                grade_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["retinopathy", "grade", "dr", "level"])]
                if name_cols and grade_cols:
                    for _, row in df.iterrows():
                        img_id = str(row[name_cols[0]]).strip()
                        grade = row[grade_cols[0]]
                        try:
                            g_int = int(grade)
                            if split != "unknown":
                                grading_labels[(split, img_id)] = g_int
                            grading_labels[("any", img_id)] = g_int
                        except (ValueError, TypeError):
                            pass
                    logger.info(f"  Found {len(df)} labels in {lf.name}")
            except Exception as e:
                logger.warning(f"  Error reading {lf.name}: {e}")

        # 4. Find images from grading directory
        image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        search_root = grading_dir / "1. Original Images" if (grading_dir / "1. Original Images").exists() else grading_dir
        candidate_images = [
            f for f in search_root.rglob("*")
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        images = []
        for img in candidate_images:
            path_lower = str(img).lower()
            if any(kw in path_lower for kw in ["mask", "groundtruth", "ground_truth", "annotation"]):
                continue
            images.append(img)

        # 5. Build records
        for img_path in images:
            stem = img_path.stem
            split = "train" if "training" in str(img_path).lower() else ("test" if "testing" in str(img_path).lower() else self._infer_split_from_path(img_path))
            dr_grade = grading_labels.get((split, stem), grading_labels.get(("any", stem)))

            num_match = re.search(r"\d+", stem)
            assigned_mask = None
            if num_match:
                num = int(num_match.group())
                matched_masks = masks_map.get((split, num), [])
                if matched_masks:
                    ma_masks = [m for m in matched_masks if "_ma" in m.name.lower()]
                    assigned_mask = str(ma_masks[0]) if ma_masks else str(matched_masks[0])
            if not assigned_mask and stem in mask_stem_index:
                assigned_mask = str(mask_stem_index[stem])

            record = {
                "image_id": f"IDRiD_{split}_{stem}",
                "image_path": str(img_path),
                "dataset_id": "IDRiD",
                "original_label": dr_grade,
                "dr_grade": dr_grade,
                "hospital_id": "eye_clinic_nanded",
                "camera_id": "kowa_vx_10alpha",
                "patient_id": None,
                "eye": None,
                "image_quality": "good",
                "gradable": True,
                "lesion_mask_path": assigned_mask,
                "lesion_annotation_type": "segmentation" if assigned_mask else None,
                "split": split,
                "duplicate_group": None,
                "label_status": "verified" if dr_grade is not None else "missing",
                "preprocessing_version": None,
            }
            records.append(record)

        return records

    def _process_aptos(self, dataset_dir: Path) -> list[dict]:
        """Process APTOS 2019 dataset into metadata records."""
        records = []

        # Find label CSV
        label_files = list(dataset_dir.rglob("*.csv"))
        labels = {}

        for lf in label_files:
            try:
                df = pd.read_csv(lf)
                id_cols = [c for c in df.columns if "id" in c.lower() or "code" in c.lower() or "image" in c.lower()]
                grade_cols = [c for c in df.columns if "diag" in c.lower() or "grade" in c.lower() or "level" in c.lower()]

                if id_cols and grade_cols:
                    for _, row in df.iterrows():
                        img_id = str(row[id_cols[0]]).strip()
                        grade = row[grade_cols[0]]
                        try:
                            labels[img_id] = int(grade)
                        except (ValueError, TypeError):
                            pass
                    logger.info(f"  Found {len(labels)} labels in {lf.name}")
            except Exception as e:
                logger.warning(f"  Error reading {lf.name}: {e}")

        # Find images
        image_extensions = {".jpg", ".jpeg", ".png"}
        images = [
            f for f in dataset_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        for img_path in images:
            stem = img_path.stem
            record = {
                "image_id": f"APTOS_{stem}",
                "image_path": str(img_path),
                "dataset_id": "APTOS",
                "original_label": labels.get(stem),
                "dr_grade": labels.get(stem),
                "hospital_id": None,
                "camera_id": None,
                "patient_id": None,
                "eye": None,
                "image_quality": None,
                "gradable": None,
                "lesion_mask_path": None,
                "lesion_annotation_type": None,
                "split": self._infer_split_from_path(img_path),
                "duplicate_group": None,
                "label_status": "verified" if stem in labels else "missing",
                "preprocessing_version": None,
            }
            records.append(record)

        return records

    def _process_ddr(self, dataset_dir: Path) -> list[dict]:
        """Process DDR dataset into metadata records."""
        records = []
        label_files = list(dataset_dir.rglob("*.csv")) + list(dataset_dir.rglob("*.txt"))
        labels = {}

        for lf in label_files:
            try:
                if lf.suffix.lower() == ".csv":
                    df = pd.read_csv(lf)
                else:
                    # DDR may use space/tab delimited text
                    df = pd.read_csv(lf, sep=r"\s+", header=None)
                    if len(df.columns) >= 2:
                        df.columns = ["image_name", "grade"] + [f"col{i}" for i in range(2, len(df.columns))]

                for col_pair in [("image_name", "grade"), ("image", "label")]:
                    if all(c in df.columns for c in col_pair):
                        for _, row in df.iterrows():
                            img_id = str(row[col_pair[0]]).strip()
                            grade = row[col_pair[1]]
                            try:
                                g = int(grade)
                                if g <= 4:  # DDR grade 5 = ungradable
                                    labels[img_id] = g
                                    labels[Path(img_id).stem] = g
                            except (ValueError, TypeError):
                                pass
            except Exception:
                pass

        # Images
        image_extensions = {".jpg", ".jpeg", ".png"}
        all_files = [
            f for f in dataset_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        images = [f for f in all_files if not any(
            kw in str(f).lower() for kw in ["mask", "segmentation", "groundtruth", "lesion"]
        )]
        masks = {f.stem: f for f in all_files if any(
            kw in str(f).lower() for kw in ["mask", "segmentation", "groundtruth", "lesion"]
        )}

        for img_path in images:
            stem = img_path.stem
            grade = labels.get(stem) or labels.get(img_path.name)
            record = {
                "image_id": f"DDR_{stem}",
                "image_path": str(img_path),
                "dataset_id": "DDR",
                "original_label": grade,
                "dr_grade": grade if grade is not None and grade <= 4 else None,
                "hospital_id": None,
                "camera_id": None,
                "patient_id": None,
                "eye": None,
                "image_quality": None,
                "gradable": grade != 5 if grade is not None else None,
                "lesion_mask_path": str(masks[stem]) if stem in masks else None,
                "lesion_annotation_type": "segmentation" if stem in masks else None,
                "split": self._infer_split_from_path(img_path),
                "duplicate_group": None,
                "label_status": "verified" if grade is not None else "missing",
                "preprocessing_version": None,
            }
            records.append(record)

        return records

    def _process_eyepacs(self, dataset_dir: Path) -> list[dict]:
        """Process EyePACS dataset into metadata records."""
        records = []
        label_files = list(dataset_dir.rglob("*.csv"))
        labels = {}

        for lf in label_files:
            try:
                df = pd.read_csv(lf)
                # EyePACS CSV: image, level
                if "image" in df.columns and "level" in df.columns:
                    for _, row in df.iterrows():
                        labels[str(row["image"]).strip()] = int(row["level"])
                    logger.info(f"  Found {len(labels)} labels in {lf.name}")
            except Exception:
                pass

        images = [
            f for f in dataset_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        for img_path in images:
            stem = img_path.stem
            record = {
                "image_id": f"EyePACS_{stem}",
                "image_path": str(img_path),
                "dataset_id": "EyePACS",
                "original_label": labels.get(stem),
                "dr_grade": labels.get(stem),
                "hospital_id": None,
                "camera_id": None,
                "patient_id": stem.split("_")[0] if "_" in stem else None,
                "eye": "left" if "_left" in stem.lower() else ("right" if "_right" in stem.lower() else None),
                "image_quality": None,
                "gradable": None,
                "lesion_mask_path": None,
                "lesion_annotation_type": None,
                "split": self._infer_split_from_path(img_path),
                "duplicate_group": None,
                "label_status": "verified" if stem in labels else "missing",
                "preprocessing_version": None,
            }
            records.append(record)

        return records

    def _process_generic(self, name: str, dataset_dir: Path) -> list[dict]:
        """Process a generic dataset with best-effort parsing."""
        records = []
        images = [
            f for f in dataset_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            and not any(kw in str(f).lower() for kw in ["mask", "segmentation", "groundtruth"])
        ]

        for img_path in images:
            records.append({
                "image_id": f"{name}_{img_path.stem}",
                "image_path": str(img_path),
                "dataset_id": name,
                "dr_grade": None,
                "original_label": None,
                "label_status": "missing",
            })

        return records

    def _infer_split_from_path(self, path: Path) -> str | None:
        """Infer train/val/test split from directory path."""
        path_lower = str(path).lower()
        if "test" in path_lower:
            return "test"
        elif "val" in path_lower:
            return "val"
        elif "train" in path_lower:
            return "train"
        return None

    def _save_schema_docs(self, df: pd.DataFrame) -> None:
        """Save schema documentation."""
        docs_dir = PROJECT_ROOT / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Dataset Schema Documentation",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nTotal records: {len(df)}",
            "",
            "## Column Descriptions",
            "",
            "| Column | Type | Description |",
            "|--------|------|-------------|",
            "| image_id | str | Stable unique identifier (dataset_stem) |",
            "| image_path | str | Path to original image |",
            "| dataset_id | str | Source dataset name |",
            "| dr_grade | int | Standardized DR grade 0-4 |",
            "| original_label | int | Label as provided by source |",
            "| hospital_id | str | Hospital identifier (if available) |",
            "| camera_id | str | Imaging device (if available) |",
            "| patient_id | str | Patient identifier (if available) |",
            "| eye | str | Left/right eye (if available) |",
            "| image_quality | float | Quality score (if computed) |",
            "| gradable | bool | Whether image is gradable |",
            "| lesion_mask_path | str | Path to lesion mask (if available) |",
            "| lesion_annotation_type | str | Type of annotation |",
            "| split | str | train/val/test/excluded |",
            "| duplicate_group | str | Duplicate group ID |",
            "| label_status | str | verified/missing/conflicting/uncertain |",
            "| preprocessing_version | str | Applied processing version |",
            "",
            "## DR Grade Mapping",
            "",
            "| Grade | Description |",
            "|-------|-------------|",
        ]
        for grade, desc in DR_GRADE_MAP.items():
            lines.append(f"| {grade} | {desc} |")

        lines.extend([
            "",
            "## Per-Dataset Summary",
            "",
        ])

        for ds_id in df["dataset_id"].unique():
            ds_df = df[df["dataset_id"] == ds_id]
            lines.append(f"### {ds_id}")
            lines.append(f"- Records: {len(ds_df)}")
            lines.append(f"- With labels: {ds_df['label_status'].eq('verified').sum()}")
            lines.append(f"- With masks: {ds_df['lesion_mask_path'].notna().sum()}")
            if ds_df["dr_grade"].notna().any():
                dist = ds_df["dr_grade"].value_counts().sort_index()
                lines.append("- Grade distribution:")
                for grade, count in dist.items():
                    lines.append(f"  - Grade {int(grade)}: {count}")
            lines.append("")

        schema_path = docs_dir / "DATASET_SCHEMA.md"
        with open(schema_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Saved schema docs: {schema_path}")


def main():
    parser = argparse.ArgumentParser(description="RetinaGuard Metadata Builder")
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()

    builder = MetadataBuilder(data_root=args.data_root)
    df = builder.build()

    if len(df) > 0:
        logger.info(f"\nMetadata Summary:")
        logger.info(f"  Total records: {len(df)}")
        logger.info(f"  Datasets: {df['dataset_id'].nunique()}")
        for ds_id in df["dataset_id"].unique():
            count = len(df[df["dataset_id"] == ds_id])
            logger.info(f"    {ds_id}: {count}")


if __name__ == "__main__":
    main()
