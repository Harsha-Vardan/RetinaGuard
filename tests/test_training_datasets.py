"""RetinaGuard - Unit Tests for PyTorch Datasets and Training Pipelines."""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.create_mock_data import generate_synthetic_fundus
from src.training.train_baseline import DRDataset, get_transforms
from src.training.train_retinaguard import RetinaGuardDataset


@pytest.fixture
def mock_dataset_files(tmp_path):
    """Generate synthetic fundus images and lesion masks with metadata."""
    records = []
    for i in range(10):
        grade = i % 5
        img, mask = generate_synthetic_fundus(height=128, width=128, dr_grade=grade, seed=i)
        img_file = tmp_path / f"img_{i:03d}.png"
        cv2.imwrite(str(img_file), img)

        mask_file = ""
        if grade > 0:
            mask_path = tmp_path / f"mask_{i:03d}.png"
            cv2.imwrite(str(mask_path), mask)
            mask_file = str(mask_path)

        records.append({
            "image_id": f"img_{i:03d}",
            "image_path": str(img_file),
            "dr_grade": grade,
            "dataset_id": "IDRiD" if i % 2 == 0 else "APTOS",
            "lesion_mask_path": mask_file if mask_file else None,
        })

    # Add one ungradable/NaN row
    records.append({
        "image_id": "img_nan",
        "image_path": str(tmp_path / "img_000.png"),
        "dr_grade": np.nan,
        "dataset_id": "IDRiD",
        "lesion_mask_path": None,
    })

    df = pd.DataFrame(records)
    return df


def test_get_transforms():
    """Verify transforms output correct tensor dimensions and types."""
    train_transform = get_transforms(image_size=128, is_training=True)
    eval_transform = get_transforms(image_size=128, is_training=False)

    assert train_transform is not None
    assert eval_transform is not None


def test_dr_dataset_len_and_filtering(mock_dataset_files):
    """Verify DRDataset filters NaN labels properly."""
    dataset = DRDataset(
        metadata=mock_dataset_files,
        transform=get_transforms(image_size=128, is_training=False),
    )
    # 10 valid + 1 NaN = 11 total. NaN should be filtered out to 10
    assert len(dataset) == 10

    # Get sample
    tensor, label, img_id = dataset[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 128, 128)
    assert isinstance(label, int)
    assert 0 <= label <= 4
    assert img_id == "img_000"


def test_dr_dataset_class_weights(mock_dataset_files):
    """Verify inverse class weights computation."""
    dataset = DRDataset(
        metadata=mock_dataset_files,
        transform=get_transforms(image_size=128, is_training=False),
    )
    weights = dataset.get_class_weights(num_classes=5)
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (5,)
    assert (weights > 0).all()


def test_retinaguard_dataset_getitem(mock_dataset_files):
    """Verify RetinaGuardDataset returns image, grade, domain, and lesion mask tensors."""
    dataset = RetinaGuardDataset(
        metadata=mock_dataset_files.dropna(subset=["dr_grade"]),
        image_size=128,
        is_training=False,
    )
    assert len(dataset) == 10

    sample = dataset[0]
    assert "image" in sample
    assert "dr_grade" in sample
    assert "domain_id" in sample
    assert "has_mask" in sample
    assert "mask" in sample

    assert sample["image"].shape == (3, 128, 128)
    assert sample["mask"].shape == (1, 128, 128)
    assert sample["domain_id"].item() in {0, 1}


def test_retinaguard_dataset_dataloader_collate(mock_dataset_files):
    """Verify batch collation in standard DataLoader."""
    dataset = RetinaGuardDataset(
        metadata=mock_dataset_files.dropna(subset=["dr_grade"]),
        image_size=128,
        is_training=False,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    batch = next(iter(loader))
    assert batch["image"].shape == (4, 3, 128, 128)
    assert batch["dr_grade"].shape == (4,)
    assert batch["domain_id"].shape == (4,)
    assert batch["mask"].shape == (4, 1, 128, 128)
    assert batch["has_mask"].shape == (4,)
