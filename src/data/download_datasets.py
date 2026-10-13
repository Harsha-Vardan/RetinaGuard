"""RetinaGuard - Dataset Download Manager.

Discovers, downloads, verifies, and extracts retinal image datasets.
Supports IDRiD (IEEE DataPort), APTOS/EyePACS (Kaggle), and DDR (manual).

Usage:
    python -m src.data.download_datasets
    python -m src.data.download_datasets --dataset IDRiD
    python -m src.data.download_datasets --dataset APTOS
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
import tarfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
import requests
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger
from src.utils.reproducibility import get_disk_space

logger = get_logger("download_datasets")


class AcquisitionStatus(str, Enum):
    """Dataset acquisition status codes."""
    NOT_FOUND = "NOT_FOUND"
    FOUND_LOCALLY = "FOUND_LOCALLY"
    DOWNLOAD_IN_PROGRESS = "DOWNLOAD_IN_PROGRESS"
    DOWNLOADED = "DOWNLOADED"
    VERIFIED = "VERIFIED"
    PARTIALLY_AVAILABLE = "PARTIALLY_AVAILABLE"
    REQUIRES_AUTHENTICATION = "REQUIRES_AUTHENTICATION"
    REQUIRES_MANUAL_APPROVAL = "REQUIRES_MANUAL_APPROVAL"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"


class DatasetDownloader:
    """Manages dataset discovery, download, verification, and extraction."""

    def __init__(self, config_path: str | Path | None = None):
        """Initialize the downloader with dataset configuration.

        Args:
            config_path: Path to datasets.yaml configuration file.
        """
        if config_path is None:
            config_path = PROJECT_ROOT / "configs" / "datasets.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.data_root = PROJECT_ROOT / self.config["project"]["data_root"]
        self.raw_dir = PROJECT_ROOT / self.config["project"]["raw_dir"]
        self.metadata_dir = PROJECT_ROOT / self.config["project"]["metadata_dir"]

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Acquisition status tracking
        self.status: dict[str, dict[str, Any]] = {}

    def run_all(self, specific_dataset: str | None = None) -> dict:
        """Run the full acquisition workflow for all or a specific dataset.

        Args:
            specific_dataset: If specified, only process this dataset.

        Returns:
            Acquisition status dictionary.
        """
        datasets = self.config["datasets"]

        if specific_dataset:
            if specific_dataset not in datasets:
                logger.error(f"Unknown dataset: {specific_dataset}. "
                           f"Available: {list(datasets.keys())}")
                return self.status
            datasets = {specific_dataset: datasets[specific_dataset]}

        # Check disk space first
        disk = get_disk_space(self.raw_dir)
        logger.info(f"Disk space available: {disk['free_gb']} GB")

        for name, ds_config in datasets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing dataset: {name} (Priority: {ds_config['priority']})")
            logger.info(f"{'='*60}")

            self.status[name] = {
                "status": AcquisitionStatus.NOT_FOUND,
                "timestamp": datetime.now().isoformat(),
                "official_url": ds_config["official_url"],
                "license": ds_config.get("license", "Unknown"),
                "local_path": None,
                "images_found": 0,
                "labels_found": False,
                "annotations_found": False,
                "errors": [],
                "user_actions_required": [],
            }

            try:
                # Step 1: Search locally
                local_path = self._search_local(name, ds_config)

                if local_path:
                    self.status[name]["status"] = AcquisitionStatus.FOUND_LOCALLY
                    self.status[name]["local_path"] = str(local_path)
                    logger.info(f"Found {name} locally at: {local_path}")
                    # Verify local data
                    self._verify_dataset(name, local_path, ds_config)
                else:
                    # Step 2: Attempt download
                    self._download_dataset(name, ds_config)

            except Exception as e:
                logger.error(f"Error processing {name}: {e}")
                self.status[name]["errors"].append(str(e))
                if self.status[name]["status"] == AcquisitionStatus.NOT_FOUND:
                    self.status[name]["status"] = AcquisitionStatus.DOWNLOAD_FAILED

        # Save status
        self._save_status()
        self._generate_report()

        return self.status

    def _search_local(self, name: str, ds_config: dict) -> Path | None:
        """Search for existing dataset files locally.

        Args:
            name: Dataset name.
            ds_config: Dataset configuration.

        Returns:
            Path to found dataset or None.
        """
        dataset_dir = self.raw_dir / name
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            return dataset_dir

        # Search for archives or existing directories matching name or patterns
        patterns = [name.lower()]
        for ap in ds_config.get("archives", []):
            pat = ap.get("pattern", "").strip("*").lower()
            if pat and pat not in patterns:
                patterns.append(pat)

        archive_exts = (".zip", ".tar.gz", ".tar", ".rar", ".7z")
        if self.raw_dir.exists():
            for item in self.raw_dir.rglob("*"):
                name_lower = item.name.lower()
                for pat in patterns:
                    if pat in name_lower:
                        if item.is_file() and any(name_lower.endswith(ext) for ext in archive_exts):
                            logger.info(f"Found archive: {item}")
                            return item
                        if item.is_dir() and any(item.iterdir()):
                            return item

        return None

    def _download_dataset(self, name: str, ds_config: dict) -> None:
        """Download a dataset based on its configured method.

        Args:
            name: Dataset name.
            ds_config: Dataset configuration.
        """
        method = ds_config.get("download_method", "manual")

        if method == "kaggle":
            self._download_kaggle(name, ds_config)
        elif method == "ieee_dataport":
            self._download_idrid(name, ds_config)
        elif method == "manual":
            self._handle_manual_download(name, ds_config)
        else:
            logger.warning(f"Unknown download method: {method}")
            self.status[name]["status"] = AcquisitionStatus.DOWNLOAD_FAILED
            self.status[name]["errors"].append(f"Unknown download method: {method}")

    def _download_kaggle(self, name: str, ds_config: dict) -> None:
        """Download dataset from Kaggle competition.

        Args:
            name: Dataset name.
            ds_config: Dataset configuration.
        """
        competition = ds_config.get("kaggle_competition")
        if not competition:
            self.status[name]["errors"].append("No kaggle_competition specified")
            self.status[name]["status"] = AcquisitionStatus.DOWNLOAD_FAILED
            return

        # Check disk space for large datasets
        est_size = ds_config.get("estimated_download_gb", 0)
        if est_size > 0:
            disk = get_disk_space(self.raw_dir)
            if disk["free_gb"] < est_size * 1.5:
                msg = (f"Insufficient disk space for {name}. "
                      f"Need ~{est_size * 1.5:.0f} GB, have {disk['free_gb']:.0f} GB")
                logger.warning(msg)
                self.status[name]["errors"].append(msg)
                self.status[name]["status"] = AcquisitionStatus.DOWNLOAD_FAILED
                self.status[name]["user_actions_required"].append(
                    f"Free up disk space or use external storage for {name}"
                )
                return

        # Check if kaggle CLI is available
        try:
            result = subprocess.run(
                ["kaggle", "--version"],
                capture_output=True, text=True, timeout=10
            )
            kaggle_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            kaggle_available = False

        if not kaggle_available:
            # Try pip install kaggle
            logger.info("Kaggle CLI not found. Attempting to install...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "kaggle", "-q"],
                    capture_output=True, text=True, timeout=60
                )
                result = subprocess.run(
                    ["kaggle", "--version"],
                    capture_output=True, text=True, timeout=10
                )
                kaggle_available = result.returncode == 0
            except Exception as e:
                logger.warning(f"Failed to install kaggle CLI: {e}")

        if not kaggle_available:
            self.status[name]["status"] = AcquisitionStatus.REQUIRES_AUTHENTICATION
            self.status[name]["user_actions_required"].extend([
                "Install kaggle CLI: pip install kaggle",
                "Create Kaggle API token: https://www.kaggle.com/settings -> API -> Create New Token",
                f"Save kaggle.json to {Path.home() / '.kaggle' / 'kaggle.json'}",
                f"Accept competition rules at: {ds_config['official_url']}",
            ])
            logger.warning(f"Kaggle CLI not available for {name}")
            return

        # Check for Kaggle credentials
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            env_user = os.environ.get("KAGGLE_USERNAME")
            env_key = os.environ.get("KAGGLE_KEY")
            if not (env_user and env_key):
                self.status[name]["status"] = AcquisitionStatus.REQUIRES_AUTHENTICATION
                self.status[name]["user_actions_required"].extend([
                    "Create Kaggle API token: https://www.kaggle.com/settings -> API -> Create New Token",
                    f"Save kaggle.json to {kaggle_json}",
                    "OR set KAGGLE_USERNAME and KAGGLE_KEY environment variables",
                    f"Accept competition rules at: {ds_config['official_url']}",
                ])
                logger.warning(f"Kaggle credentials not found for {name}")
                return

        # Download
        dataset_dir = self.raw_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {name} from Kaggle competition: {competition}")
        self.status[name]["status"] = AcquisitionStatus.DOWNLOAD_IN_PROGRESS

        try:
            result = subprocess.run(
                [
                    "kaggle", "competitions", "download",
                    "-c", competition,
                    "-p", str(dataset_dir),
                ],
                capture_output=True, text=True, timeout=3600
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip()
                if "403" in error_msg or "accept" in error_msg.lower():
                    self.status[name]["status"] = AcquisitionStatus.REQUIRES_MANUAL_APPROVAL
                    self.status[name]["user_actions_required"].append(
                        f"Accept competition rules at: {ds_config['official_url']}/rules"
                    )
                else:
                    self.status[name]["status"] = AcquisitionStatus.DOWNLOAD_FAILED
                    self.status[name]["errors"].append(error_msg)
                logger.error(f"Kaggle download failed for {name}: {error_msg}")
                return

            self.status[name]["status"] = AcquisitionStatus.DOWNLOADED
            logger.info(f"Downloaded {name} to {dataset_dir}")

            # Extract archives
            self._extract_archives(dataset_dir)

            # Verify
            self._verify_dataset(name, dataset_dir, ds_config)

        except subprocess.TimeoutExpired:
            self.status[name]["status"] = AcquisitionStatus.DOWNLOAD_FAILED
            self.status[name]["errors"].append("Download timed out (1 hour limit)")
        except Exception as e:
            self.status[name]["status"] = AcquisitionStatus.DOWNLOAD_FAILED
            self.status[name]["errors"].append(str(e))

    def _download_idrid(self, name: str, ds_config: dict) -> None:
        """Handle IDRiD dataset acquisition.

        IDRiD is hosted on IEEE DataPort. Direct automated download requires
        IEEE DataPort access. We attempt multiple approaches.

        Args:
            name: Dataset name.
            ds_config: Dataset configuration.
        """
        dataset_dir = self.raw_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info("IDRiD dataset is hosted on IEEE DataPort.")
        logger.info("Checking for alternative authorized download sources...")

        # Attempt 1: Check if kaggle has IDRiD mirror
        try:
            result = subprocess.run(
                ["kaggle", "datasets", "list", "-s", "idrid"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and "idrid" in result.stdout.lower():
                logger.info("Found IDRiD-related datasets on Kaggle.")
                # Try to download known IDRiD mirrors
                kaggle_datasets = [
                    "aaryapatel98/indian-diabetic-retinopathy-image-dataset-idrid",
                ]
                for kds in kaggle_datasets:
                    try:
                        dl_result = subprocess.run(
                            [
                                "kaggle", "datasets", "download",
                                "-d", kds,
                                "-p", str(dataset_dir),
                            ],
                            capture_output=True, text=True, timeout=600
                        )
                        if dl_result.returncode == 0:
                            logger.info(f"Downloaded IDRiD from Kaggle: {kds}")
                            self.status[name]["status"] = AcquisitionStatus.DOWNLOADED
                            self._extract_archives(dataset_dir)
                            self._verify_dataset(name, dataset_dir, ds_config)
                            return
                    except Exception:
                        continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # If Kaggle method failed, provide manual instructions
        self.status[name]["status"] = AcquisitionStatus.REQUIRES_MANUAL_APPROVAL
        self.status[name]["user_actions_required"] = [
            "IDRiD dataset requires manual download from IEEE DataPort:",
            f"  1. Visit: {ds_config['official_url']}",
            "  2. Sign in to IEEE DataPort (free account required)",
            "  3. Download the dataset archives (A. Disease Grading, B. Localization, C. Segmentation)",
            f"  4. Place downloaded archives in: {dataset_dir}",
            "  5. Re-run: python -m src.data.download_datasets --dataset IDRiD",
            "",
            "Alternative: Check Kaggle for authorized mirrors:",
            "  kaggle datasets list -s idrid",
        ]
        logger.warning("IDRiD requires manual download. See instructions above.")

    def _handle_manual_download(self, name: str, ds_config: dict) -> None:
        """Handle datasets that require manual download.

        Args:
            name: Dataset name.
            ds_config: Dataset configuration.
        """
        self.status[name]["status"] = AcquisitionStatus.REQUIRES_MANUAL_APPROVAL
        self.status[name]["user_actions_required"] = [
            f"Dataset {name} requires manual download:",
            f"  Official source: {ds_config['official_url']}",
            f"  Place files in: {self.raw_dir / name}",
            "  Re-run download_datasets after placing the files.",
        ]
        logger.warning(f"{name} requires manual download from: {ds_config['official_url']}")

    def _extract_archives(self, directory: Path) -> None:
        """Extract all archive files in a directory.

        Args:
            directory: Directory containing archives.
        """
        for archive in directory.glob("*.zip"):
            logger.info(f"Extracting: {archive.name}")
            try:
                with zipfile.ZipFile(archive, "r") as zf:
                    # Check for corrupted archive
                    bad_file = zf.testzip()
                    if bad_file:
                        logger.warning(f"Corrupted file in archive: {bad_file}")

                    zf.extractall(directory)
                logger.info(f"Extracted: {archive.name}")
            except zipfile.BadZipFile:
                logger.error(f"Bad zip file: {archive.name}")

        for archive in directory.glob("*.tar.gz"):
            logger.info(f"Extracting: {archive.name}")
            try:
                with tarfile.open(archive, "r:gz") as tf:
                    tf.extractall(directory, filter="data")
                logger.info(f"Extracted: {archive.name}")
            except tarfile.TarError as e:
                logger.error(f"Error extracting {archive.name}: {e}")

        for archive in directory.glob("*.tar"):
            if not str(archive).endswith(".tar.gz"):
                logger.info(f"Extracting: {archive.name}")
                try:
                    with tarfile.open(archive, "r") as tf:
                        tf.extractall(directory, filter="data")
                    logger.info(f"Extracted: {archive.name}")
                except tarfile.TarError as e:
                    logger.error(f"Error extracting {archive.name}: {e}")

    def _verify_dataset(self, name: str, path: Path, ds_config: dict) -> None:
        """Verify a dataset's contents.

        Args:
            name: Dataset name.
            path: Path to dataset.
            ds_config: Dataset configuration.
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        label_extensions = {".csv", ".xlsx", ".xls"}
        mask_extensions = {".jpg", ".png", ".tif", ".tiff", ".bmp"}

        images = []
        labels = []
        masks = []
        annotation_files = []

        for f in path.rglob("*"):
            if f.is_file():
                suffix = f.suffix.lower()
                if suffix in image_extensions:
                    # Distinguish between images and masks
                    path_lower = str(f).lower()
                    if any(kw in path_lower for kw in
                           ["mask", "segmentation", "annotation", "label_map",
                            "lesion", "groundtruth", "ground_truth"]):
                        masks.append(f)
                    else:
                        images.append(f)
                elif suffix in label_extensions:
                    labels.append(f)
                elif suffix == ".json":
                    annotation_files.append(f)

        self.status[name]["images_found"] = len(images)
        self.status[name]["labels_found"] = len(labels) > 0
        self.status[name]["label_files"] = [str(l) for l in labels]
        self.status[name]["annotations_found"] = len(masks) > 0 or len(annotation_files) > 0
        self.status[name]["masks_found"] = len(masks)
        self.status[name]["annotation_files"] = len(annotation_files)

        # Log summary
        logger.info(f"  Images: {len(images)}")
        logger.info(f"  Label files: {len(labels)}")
        logger.info(f"  Mask files: {len(masks)}")
        logger.info(f"  Annotation files: {len(annotation_files)}")

        if len(images) > 0 and len(labels) > 0:
            if self.status[name]["status"] != AcquisitionStatus.VERIFIED:
                self.status[name]["status"] = AcquisitionStatus.VERIFIED
        elif len(images) > 0:
            self.status[name]["status"] = AcquisitionStatus.PARTIALLY_AVAILABLE
            self.status[name]["errors"].append("Images found but no label files")
        else:
            if self.status[name]["status"] == AcquisitionStatus.FOUND_LOCALLY:
                self.status[name]["status"] = AcquisitionStatus.PARTIALLY_AVAILABLE
                self.status[name]["errors"].append("Dataset directory exists but no images found")

        # Count by subdirectory
        subdirs = {}
        for img in images:
            parent = img.parent.name
            subdirs[parent] = subdirs.get(parent, 0) + 1
        self.status[name]["subdirectories"] = subdirs

        self.status[name]["local_path"] = str(path)

    def _save_status(self) -> None:
        """Save acquisition status to JSON."""
        status_path = self.metadata_dir / "dataset_acquisition_status.json"
        # Convert enums to strings
        serializable = {}
        for name, info in self.status.items():
            serializable[name] = {
                k: (v.value if isinstance(v, AcquisitionStatus) else v)
                for k, v in info.items()
            }

        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info(f"Saved acquisition status to {status_path}")

    def _generate_report(self) -> None:
        """Generate a human-readable acquisition report."""
        docs_dir = PROJECT_ROOT / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        report_path = docs_dir / "DATASET_ACQUISITION_REPORT.md"

        lines = [
            "# Dataset Acquisition Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        for name, info in self.status.items():
            status_val = info["status"]
            if isinstance(status_val, AcquisitionStatus):
                status_val = status_val.value

            status_emoji = {
                "VERIFIED": "✅",
                "FOUND_LOCALLY": "📂",
                "DOWNLOADED": "📥",
                "PARTIALLY_AVAILABLE": "⚠️",
                "REQUIRES_AUTHENTICATION": "🔐",
                "REQUIRES_MANUAL_APPROVAL": "📋",
                "DOWNLOAD_FAILED": "❌",
                "NOT_FOUND": "❓",
                "DOWNLOAD_IN_PROGRESS": "⏳",
            }.get(status_val, "❓")

            lines.extend([
                f"## {name} {status_emoji}",
                f"- **Status**: `{status_val}`",
                f"- **Official URL**: {info.get('official_url', 'N/A')}",
                f"- **License**: {info.get('license', 'Unknown')}",
                f"- **Local path**: `{info.get('local_path', 'None')}`",
                f"- **Images found**: {info.get('images_found', 0)}",
                f"- **Labels found**: {info.get('labels_found', False)}",
                f"- **Annotations found**: {info.get('annotations_found', False)}",
            ])

            if info.get("masks_found"):
                lines.append(f"- **Masks found**: {info['masks_found']}")

            if info.get("subdirectories"):
                lines.append("- **Subdirectories**:")
                for subdir, count in info["subdirectories"].items():
                    lines.append(f"  - `{subdir}`: {count} images")

            if info.get("errors"):
                lines.append("- **Errors**:")
                for error in info["errors"]:
                    lines.append(f"  - {error}")

            if info.get("user_actions_required"):
                lines.append("- **Required actions**:")
                for action in info["user_actions_required"]:
                    lines.append(f"  - {action}")

            lines.append("")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Generated acquisition report: {report_path}")


def main():
    """Main entry point for dataset download."""
    parser = argparse.ArgumentParser(
        description="RetinaGuard Dataset Download Manager"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Specific dataset to process (e.g., IDRiD, APTOS, DDR, EyePACS)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to datasets.yaml configuration file"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RetinaGuard Dataset Download Manager")
    logger.info("=" * 60)

    downloader = DatasetDownloader(config_path=args.config)
    status = downloader.run_all(specific_dataset=args.dataset)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ACQUISITION SUMMARY")
    logger.info("=" * 60)
    for name, info in status.items():
        status_val = info["status"]
        if isinstance(status_val, AcquisitionStatus):
            status_val = status_val.value
        logger.info(f"  {name}: {status_val}")


if __name__ == "__main__":
    main()
