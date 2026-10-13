"""RetinaGuard - Unit tests for data splitting."""

import pandas as pd
import pytest

from src.data.split_data import create_splits


@pytest.fixture
def dummy_metadata_csv(tmp_path):
    """Create a temporary dummy metadata CSV."""
    data = {
        "image_id": [f"img_{i:03d}" for i in range(30)],
        "dr_grade": [i % 5 for i in range(30)],
        "patient_id": [f"patient_{i // 2:02d}" for i in range(30)],
        "label_status": ["verified"] * 28 + ["excluded", "excluded"],
        "split": [None] * 30,
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "metadata.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_create_splits(dummy_metadata_csv, tmp_path):
    """Verify train/val/test splits generation and preservation of exclusions."""
    out_dir = tmp_path / "manifests"
    df_splits = create_splits(
        metadata_path=dummy_metadata_csv,
        output_dir=out_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    assert "split" in df_splits.columns

    # Excluded records must be marked as excluded
    excluded_count = (df_splits["split"] == "excluded").sum()
    assert excluded_count == 2

    # Manifests should exist
    assert (out_dir / "train_manifest.csv").exists()
    assert (out_dir / "val_manifest.csv").exists()
    assert (out_dir / "test_manifest.csv").exists()

    train_df = pd.read_csv(out_dir / "train_manifest.csv")
    val_df = pd.read_csv(out_dir / "val_manifest.csv")
    test_df = pd.read_csv(out_dir / "test_manifest.csv")

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert len(train_df) + len(val_df) + len(test_df) == 28
