"""RetinaGuard - Image Preprocessing Pipeline.

Implements retinal image preprocessing: FOV detection, border removal,
resizing, conservative normalization, and quality assessment.

Usage:
    python -m src.data.preprocessing
    python -m src.data.preprocessing --config configs/preprocessing.yaml
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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger("preprocessing")


class RetinalPreprocessor:
    """Preprocesses retinal fundus images for DR classification."""

    def __init__(self, config: dict | None = None, config_path: str | Path | None = None):
        """Initialize with configuration.

        Args:
            config: Preprocessing configuration dictionary.
            config_path: Path to preprocessing.yaml.
        """
        if config is None and config_path is not None:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)["preprocessing"]
        elif config is None:
            config_path = PROJECT_ROOT / "configs" / "preprocessing.yaml"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)["preprocessing"]
            else:
                config = self._default_config()

        self.config = config
        self.version = config.get("version", "1.0")

    @staticmethod
    def _default_config() -> dict:
        return {
            "version": "1.0",
            "fov_detection": {"enabled": True, "threshold": 10, "margin": 5},
            "resize": {"enabled": True, "target_size": [512, 512],
                      "preserve_aspect_ratio": True, "padding_color": [0, 0, 0]},
            "normalization": {"enabled": True, "method": "clahe",
                            "clahe": {"clip_limit": 2.0, "tile_grid_size": [8, 8]},
                            "color_space": "LAB"},
            "quality": {"enabled": True, "min_dimension": 100,
                       "blur_threshold": 50.0, "dark_threshold": 30,
                       "bright_threshold": 240},
        }

    def process_image(
        self, image_path: str | Path, save_path: str | Path | None = None,
        mask_path: str | Path | None = None, mask_save_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Process a single retinal image.

        Args:
            image_path: Path to input image.
            save_path: Path to save processed image. If None, returns without saving.
            mask_path: Optional path to corresponding lesion mask.
            mask_save_path: Path to save processed mask.

        Returns:
            Processing metadata dictionary.
        """
        image_path = Path(image_path)
        meta = {
            "original_path": str(image_path),
            "processed_path": str(save_path) if save_path else None,
            "status": "pending",
            "quality_flags": [],
            "original_size": None,
            "processed_size": None,
            "preprocessing_version": self.version,
        }

        # Step 1: Load and validate
        img = cv2.imread(str(image_path))
        if img is None:
            meta["status"] = "failed"
            meta["quality_flags"].append("unreadable")
            return meta

        h, w = img.shape[:2]
        meta["original_size"] = [w, h]

        # Quality check on original
        quality = self._assess_quality(img)
        meta["quality_flags"] = quality["flags"]

        if quality.get("min_dim_fail"):
            meta["status"] = "excluded"
            meta["quality_flags"].append("too_small")
            return meta

        # Load mask if provided
        mask = None
        if mask_path and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Step 2: FOV detection and cropping
        if self.config["fov_detection"]["enabled"]:
            img, crop_info = self._detect_and_crop_fov(img)
            if mask is not None:
                mask = mask[
                    crop_info["y1"]:crop_info["y2"],
                    crop_info["x1"]:crop_info["x2"]
                ]

        # Step 3: Resize
        if self.config["resize"]["enabled"]:
            target = self.config["resize"]["target_size"]
            img, mask = self._resize_with_aspect(img, target, mask)

        # Step 4: Normalization
        if self.config["normalization"]["enabled"]:
            img = self._normalize(img)

        meta["processed_size"] = [img.shape[1], img.shape[0]]
        meta["status"] = "processed"

        # Save
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), img)
            meta["processed_path"] = str(save_path)

        if mask is not None and mask_save_path:
            mask_save_path = Path(mask_save_path)
            mask_save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_save_path), mask)

        return meta

    def _detect_and_crop_fov(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        """Detect the retinal field of view and crop to it.

        Uses grayscale thresholding to find the retinal region and
        crops to its bounding box with a small margin.
        """
        threshold = self.config["fov_detection"]["threshold"]
        margin = self.config["fov_detection"]["margin"]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Find bounding box of non-black region
        coords = cv2.findNonZero(binary)
        if coords is None:
            # Entire image is dark — return as-is
            h, w = img.shape[:2]
            return img, {"x1": 0, "y1": 0, "x2": w, "y2": h}

        x, y, w, h = cv2.boundingRect(coords)

        # Apply margin
        img_h, img_w = img.shape[:2]
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_w, x + w + margin)
        y2 = min(img_h, y + h + margin)

        cropped = img[y1:y2, x1:x2]
        return cropped, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def _resize_with_aspect(
        self, img: np.ndarray, target_size: list[int],
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Resize image preserving aspect ratio with padding.

        Args:
            img: Input image.
            target_size: [height, width].
            mask: Optional mask to resize.

        Returns:
            Resized image and mask.
        """
        target_h, target_w = target_size
        h, w = img.shape[:2]

        if self.config["resize"].get("preserve_aspect_ratio", True):
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Resize
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

            # Pad
            pad_color = self.config["resize"].get("padding_color", [0, 0, 0])
            canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            img = canvas

            if mask is not None:
                resized_mask = cv2.resize(
                    mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
                )
                mask_canvas = np.zeros((target_h, target_w), dtype=np.uint8)
                mask_canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_mask
                mask = mask_canvas
        else:
            img = cv2.resize(img, (target_w, target_h))
            if mask is not None:
                mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        return img, mask

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Apply conservative illumination normalization.

        Uses CLAHE on the L channel in LAB color space to preserve
        color relationships while improving contrast.
        """
        method = self.config["normalization"]["method"]

        if method == "clahe":
            clahe_config = self.config["normalization"].get("clahe", {})
            clip_limit = clahe_config.get("clip_limit", 2.0)
            tile_size = tuple(clahe_config.get("tile_grid_size", [8, 8]))

            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            l_channel = clahe.apply(l_channel)

            lab = cv2.merge([l_channel, a_channel, b_channel])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return img

    def _assess_quality(self, img: np.ndarray) -> dict:
        """Assess image quality with basic metrics.

        Returns:
            Dictionary with quality metrics and flags.
        """
        quality_config = self.config.get("quality", {})
        flags = []
        h, w = img.shape[:2]

        result = {
            "height": h,
            "width": w,
            "flags": flags,
        }

        # Minimum dimension
        min_dim = quality_config.get("min_dimension", 100)
        if min(h, w) < min_dim:
            flags.append("too_small")
            result["min_dim_fail"] = True

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur detection (Laplacian variance)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        result["blur_score"] = float(lap_var)
        if lap_var < quality_config.get("blur_threshold", 50.0):
            flags.append("blurry")

        # Exposure
        mean_intensity = float(gray.mean())
        result["mean_intensity"] = mean_intensity
        if mean_intensity < quality_config.get("dark_threshold", 30):
            flags.append("too_dark")
        if mean_intensity > quality_config.get("bright_threshold", 240):
            flags.append("too_bright")

        return result

    def process_dataset(
        self,
        metadata_path: str | Path,
        output_dir: str | Path,
    ) -> pd.DataFrame:
        """Process all images listed in a metadata CSV.

        Args:
            metadata_path: Path to master_metadata.csv.
            output_dir: Directory for processed images.

        Returns:
            DataFrame with processing results.
        """
        metadata_path = Path(metadata_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(metadata_path)
        logger.info(f"Processing {len(df)} images...")

        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
            image_path = row["image_path"]
            if not Path(image_path).exists():
                results.append({
                    "image_id": row["image_id"],
                    "status": "file_not_found",
                })
                continue

            # Create output path
            dataset_id = row.get("dataset_id", "unknown")
            img_name = Path(image_path).name
            save_path = output_dir / dataset_id / img_name

            # Process mask if available
            mask_path = row.get("lesion_mask_path")
            mask_save = None
            if pd.notna(mask_path) and Path(mask_path).exists():
                mask_save = output_dir / dataset_id / "masks" / Path(mask_path).name

            result = self.process_image(
                image_path, save_path, mask_path if pd.notna(mask_path) else None, mask_save
            )
            result["image_id"] = row["image_id"]
            results.append(result)

        results_df = pd.DataFrame(results)

        # Save processing report
        report_path = output_dir / "preprocessing_report.csv"
        results_df.to_csv(report_path, index=False)

        # Summary
        status_counts = results_df["status"].value_counts()
        logger.info(f"\nPreprocessing Summary:")
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count}")

        return results_df


def main():
    parser = argparse.ArgumentParser(description="RetinaGuard Image Preprocessor")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to preprocessing.yaml")
    parser.add_argument("--metadata", type=str,
                       default="data/metadata/master_metadata.csv",
                       help="Path to metadata CSV")
    parser.add_argument("--output", type=str, default="data/processed",
                       help="Output directory for processed images")
    args = parser.parse_args()

    preprocessor = RetinalPreprocessor(config_path=args.config)
    metadata_path = PROJECT_ROOT / args.metadata
    output_dir = PROJECT_ROOT / args.output

    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        logger.info("Run 'python -m src.data.build_metadata' first.")
        sys.exit(1)

    preprocessor.process_dataset(metadata_path, output_dir)


if __name__ == "__main__":
    main()
