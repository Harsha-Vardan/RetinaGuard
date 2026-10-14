"""RetinaGuard - Unit Tests for MetadataBuilder, DatasetDownloader, and Reproducibility."""

from pathlib import Path
import pandas as pd
import pytest
import yaml

from src.data.build_metadata import MetadataBuilder
from src.data.download_datasets import AcquisitionStatus, DatasetDownloader
from src.utils.reproducibility import get_hardware_info, save_experiment_config


def test_metadata_builder_initialization():
    """Verify MetadataBuilder initializes paths correctly."""
    builder = MetadataBuilder()
    assert builder.raw_dir.name == "raw"
    assert builder.metadata_dir.name == "metadata"


def test_metadata_builder_infer_split_from_path():
    """Verify split inference from directory strings."""
    builder = MetadataBuilder()
    assert builder._infer_split_from_path(Path("data/raw/IDRiD/train/img_01.jpg")) == "train"
    assert builder._infer_split_from_path(Path("data/raw/IDRiD/test/img_01.jpg")) == "test"
    assert builder._infer_split_from_path(Path("data/raw/IDRiD/val/img_01.jpg")) == "val"
    assert builder._infer_split_from_path(Path("data/raw/IDRiD/images/img_01.jpg")) is None


def test_acquisition_status_enum():
    """Verify all defined acquisition status codes are present."""
    expected_statuses = [
        "NOT_FOUND",
        "FOUND_LOCALLY",
        "DOWNLOAD_IN_PROGRESS",
        "DOWNLOADED",
        "VERIFIED",
        "PARTIALLY_AVAILABLE",
        "REQUIRES_AUTHENTICATION",
        "REQUIRES_MANUAL_APPROVAL",
        "DOWNLOAD_FAILED",
    ]
    for status_str in expected_statuses:
        status_enum = AcquisitionStatus(status_str)
        assert status_enum.value == status_str


def test_dataset_downloader_init_default_config():
    """Verify DatasetDownloader loads datasets.yaml config."""
    downloader = DatasetDownloader()
    assert isinstance(downloader.config, dict)
    assert "datasets" in downloader.config or len(downloader.config) > 0


def test_reproducibility_hardware_info():
    """Verify get_hardware_info returns complete platform information."""
    hw_info = get_hardware_info()
    assert "platform" in hw_info
    assert "python_version" in hw_info
    assert "pytorch_version" in hw_info
    assert "cuda_available" in hw_info
    assert "cpu_count" in hw_info


def test_save_experiment_config_roundtrip(tmp_path):
    """Verify save_experiment_config writes configuration JSON that can be reloaded."""
    import json
    dummy_config = {
        "model": {"architecture": "resnet18", "num_classes": 5},
        "training": {"batch_size": 32, "epochs": 20, "lr": 1e-4},
        "dataset": {"name": "IDRiD_APTOS_Unified"},
    }
    out_dir = tmp_path / "exp_01"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = save_experiment_config(dummy_config, out_dir)
    assert config_path.exists()

    with open(config_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["config"]["model"]["architecture"] == "resnet18"
    assert loaded["config"]["training"]["epochs"] == 20
    assert "hardware" in loaded
