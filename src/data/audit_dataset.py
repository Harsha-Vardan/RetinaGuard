"""RetinaGuard - Dataset Auditing.

Inspects dataset contents, validates images/labels, detects issues,
and generates comprehensive audit reports.

Usage:
    python -m src.data.audit_dataset
    python -m src.data.audit_dataset --dataset IDRiD
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
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

logger = get_logger("audit_dataset")


class DatasetAuditor:
    """Audits retinal image datasets for quality and integrity."""

    VALID_DR_GRADES = {0, 1, 2, 3, 4}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    def __init__(self, data_root: str | Path | None = None):
        self.data_root = Path(data_root or PROJECT_ROOT / "data")
        self.raw_dir = self.data_root / "raw"
        self.metadata_dir = self.data_root / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = PROJECT_ROOT / "outputs" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def audit_all(self, specific_dataset: str | None = None) -> dict:
        """Run audit on all discovered datasets or a specific one.

        Args:
            specific_dataset: If specified, only audit this dataset.

        Returns:
            Audit results dictionary.
        """
        results = {}

        if specific_dataset:
            dataset_path = self.raw_dir / specific_dataset
            if dataset_path.exists():
                results[specific_dataset] = self._audit_dataset(
                    specific_dataset, dataset_path
                )
            else:
                logger.error(f"Dataset directory not found: {dataset_path}")
        else:
            for item in self.raw_dir.iterdir():
                if item.is_dir():
                    results[item.name] = self._audit_dataset(item.name, item)

        # Generate reports
        self._save_audit_report(results)
        return results

    def _audit_dataset(self, name: str, path: Path) -> dict:
        """Audit a single dataset.

        Args:
            name: Dataset name.
            path: Path to dataset directory.

        Returns:
            Audit result dictionary.
        """
        logger.info(f"\nAuditing dataset: {name}")
        logger.info(f"Path: {path}")

        audit = {
            "dataset": name,
            "path": str(path),
            "timestamp": datetime.now().isoformat(),
            "images": {"total": 0, "valid": 0, "corrupted": [], "dimensions": {}},
            "labels": {"found": False, "files": [], "grade_distribution": {}, "invalid": []},
            "annotations": {"masks": 0, "annotation_files": 0, "coverage": {}},
            "quality": {"blur_flagged": 0, "dark_flagged": 0, "bright_flagged": 0},
            "duplicates": {"exact": 0, "groups": []},
            "issues": [],
        }

        # Discover all files
        all_files = list(path.rglob("*"))
        all_files = [f for f in all_files if f.is_file()]
        logger.info(f"Total files found: {len(all_files)}")

        # Categorize files
        images = []
        label_files = []
        mask_files = []
        other_files = []

        for f in all_files:
            suffix = f.suffix.lower()
            path_lower = str(f).lower()
            if suffix in self.IMAGE_EXTENSIONS:
                if any(kw in path_lower for kw in [
                    "mask", "segmentation", "groundtruth", "ground_truth",
                    "groundtruths", "lesion"
                ]):
                    mask_files.append(f)
                else:
                    images.append(f)
            elif suffix in {".csv", ".xlsx", ".xls"}:
                label_files.append(f)
            else:
                other_files.append(f)

        audit["images"]["total"] = len(images)
        audit["annotations"]["masks"] = len(mask_files)
        audit["labels"]["files"] = [str(f) for f in label_files]

        # Audit images
        logger.info(f"Auditing {len(images)} images...")
        hashes = {}  # For duplicate detection
        dim_counter = Counter()

        for img_path in tqdm(images, desc=f"Checking {name} images", disable=len(images) < 10):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    audit["images"]["corrupted"].append(str(img_path))
                    continue

                audit["images"]["valid"] += 1
                h, w, c = img.shape
                dim_counter[f"{w}x{h}x{c}"] += 1

                # Quick quality checks
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                mean_intensity = gray.mean()

                if lap_var < 50.0:
                    audit["quality"]["blur_flagged"] += 1
                if mean_intensity < 30:
                    audit["quality"]["dark_flagged"] += 1
                if mean_intensity > 240:
                    audit["quality"]["bright_flagged"] += 1

                # Hash for duplicate detection
                file_hash = hashlib.md5(img_path.read_bytes()).hexdigest()
                if file_hash in hashes:
                    audit["duplicates"]["exact"] += 1
                    audit["duplicates"]["groups"].append(
                        [str(hashes[file_hash]), str(img_path)]
                    )
                else:
                    hashes[file_hash] = img_path

            except Exception as e:
                audit["images"]["corrupted"].append(str(img_path))
                audit["issues"].append(f"Error reading {img_path.name}: {e}")

        audit["images"]["dimensions"] = dict(dim_counter)

        # Audit labels
        for lf in label_files:
            try:
                if lf.suffix.lower() == ".csv":
                    df = pd.read_csv(lf)
                else:
                    df = pd.read_excel(lf)

                logger.info(f"Label file: {lf.name}, columns: {list(df.columns)}, rows: {len(df)}")
                audit["labels"]["found"] = True

                # Try to find DR grade column
                grade_cols = [c for c in df.columns if any(
                    kw in c.lower() for kw in
                    ["grade", "level", "diagnosis", "retinopathy", "dr", "label"]
                )]
                if grade_cols:
                    col = grade_cols[0]
                    grades = df[col].dropna()
                    dist = grades.value_counts().to_dict()
                    audit["labels"]["grade_distribution"][lf.name] = {
                        str(k): int(v) for k, v in dist.items()
                    }

                    # Check for invalid grades
                    for grade in grades.unique():
                        try:
                            g = int(grade)
                            if g not in self.VALID_DR_GRADES and g != 5:
                                audit["labels"]["invalid"].append(
                                    f"{lf.name}: unexpected grade {g}"
                                )
                        except (ValueError, TypeError):
                            audit["labels"]["invalid"].append(
                                f"{lf.name}: non-numeric grade '{grade}'"
                            )

            except Exception as e:
                audit["issues"].append(f"Error reading label file {lf.name}: {e}")

        # Audit masks
        if mask_files:
            logger.info(f"Found {len(mask_files)} mask files")
            # Check alignment with images
            mask_stems = {m.stem for m in mask_files}
            image_stems = {i.stem for i in images}
            matched = mask_stems & image_stems
            unmatched_masks = mask_stems - image_stems
            images_without_masks = image_stems - mask_stems

            audit["annotations"]["coverage"] = {
                "matched_pairs": len(matched),
                "unmatched_masks": len(unmatched_masks),
                "images_without_masks": len(images_without_masks),
            }

        # Summarize issues
        if audit["images"]["corrupted"]:
            audit["issues"].append(
                f"{len(audit['images']['corrupted'])} corrupted images"
            )
        if audit["duplicates"]["exact"] > 0:
            audit["issues"].append(
                f"{audit['duplicates']['exact']} exact duplicate images"
            )
        if not audit["labels"]["found"]:
            audit["issues"].append("No label files found")

        logger.info(f"Audit complete for {name}:")
        logger.info(f"  Valid images: {audit['images']['valid']}/{audit['images']['total']}")
        logger.info(f"  Labels found: {audit['labels']['found']}")
        logger.info(f"  Masks: {audit['annotations']['masks']}")
        logger.info(f"  Issues: {len(audit['issues'])}")

        return audit

    def _save_audit_report(self, results: dict) -> None:
        """Save audit results to JSON and Markdown."""
        # JSON report
        audit_path = self.reports_dir / "dataset_audit.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved audit JSON: {audit_path}")

        # Class distribution CSV
        dist_rows = []
        for ds_name, audit in results.items():
            for label_file, dist in audit["labels"]["grade_distribution"].items():
                for grade, count in dist.items():
                    dist_rows.append({
                        "dataset": ds_name,
                        "label_file": label_file,
                        "grade": grade,
                        "count": count,
                    })
        if dist_rows:
            dist_path = self.reports_dir / "class_distribution.csv"
            pd.DataFrame(dist_rows).to_csv(dist_path, index=False)
            logger.info(f"Saved class distribution: {dist_path}")

        # Duplicate report
        dup_rows = []
        for ds_name, audit in results.items():
            for group in audit["duplicates"]["groups"]:
                for path in group:
                    dup_rows.append({"dataset": ds_name, "file_path": path})
        if dup_rows:
            dup_path = self.reports_dir / "duplicate_report.csv"
            pd.DataFrame(dup_rows).to_csv(dup_path, index=False)

        # Markdown report
        docs_dir = PROJECT_ROOT / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        md_path = docs_dir / "DATASET_AUDIT.md"

        lines = [
            "# Dataset Audit Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        for ds_name, audit in results.items():
            lines.extend([
                f"## {ds_name}",
                f"- Path: `{audit['path']}`",
                f"- Total images: {audit['images']['total']}",
                f"- Valid images: {audit['images']['valid']}",
                f"- Corrupted: {len(audit['images']['corrupted'])}",
                f"- Image dimensions: {audit['images']['dimensions']}",
                f"- Label files found: {audit['labels']['found']}",
                f"- Mask files: {audit['annotations']['masks']}",
                f"- Exact duplicates: {audit['duplicates']['exact']}",
                f"- Quality flags: blur={audit['quality']['blur_flagged']}, "
                f"dark={audit['quality']['dark_flagged']}, "
                f"bright={audit['quality']['bright_flagged']}",
            ])

            if audit["labels"]["grade_distribution"]:
                lines.append("\n### Grade Distribution")
                for lf, dist in audit["labels"]["grade_distribution"].items():
                    lines.append(f"\n**{lf}**:")
                    lines.append("| Grade | Count |")
                    lines.append("|-------|-------|")
                    for grade, count in sorted(dist.items(), key=lambda x: str(x[0])):
                        lines.append(f"| {grade} | {count} |")

            if audit.get("annotations", {}).get("coverage"):
                cov = audit["annotations"]["coverage"]
                lines.extend([
                    "\n### Annotation Coverage",
                    f"- Matched image-mask pairs: {cov.get('matched_pairs', 0)}",
                    f"- Unmatched masks: {cov.get('unmatched_masks', 0)}",
                    f"- Images without masks: {cov.get('images_without_masks', 0)}",
                ])

            if audit["issues"]:
                lines.append("\n### Issues")
                for issue in audit["issues"]:
                    lines.append(f"- ⚠️ {issue}")
            lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Saved audit report: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="RetinaGuard Dataset Auditor")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Specific dataset to audit")
    parser.add_argument("--data-root", type=str, default=None,
                       help="Data root directory")
    args = parser.parse_args()

    auditor = DatasetAuditor(data_root=args.data_root)
    auditor.audit_all(specific_dataset=args.dataset)


if __name__ == "__main__":
    main()
