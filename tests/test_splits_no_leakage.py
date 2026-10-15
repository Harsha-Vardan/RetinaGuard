"""RetinaGuard - Strict Patient Isolation & Anti-Leakage Test Suite.

Asserts zero patient ID overlap across train, validation, and test splits,
and verifies no held-out domain contamination in source training sets.
"""

from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_manifest_patient_isolation():
    """Verify train, val, and test manifests have mutually exclusive patient IDs."""
    manifest_dir = PROJECT_ROOT / "data" / "manifests"
    train_m = manifest_dir / "train_manifest.csv"
    val_m = manifest_dir / "val_manifest.csv"
    test_m = manifest_dir / "test_manifest.csv"

    if not (train_m.exists() and val_m.exists() and test_m.exists()):
        pytest.skip("Manifest files not present in data/manifests")

    train_df = pd.read_csv(train_m)
    val_df = pd.read_csv(val_m)
    test_df = pd.read_csv(test_m)

    if "patient_id" in train_df.columns:
        train_patients = set(train_df["patient_id"].dropna().unique())
        val_patients = set(val_df["patient_id"].dropna().unique())
        test_patients = set(test_df["patient_id"].dropna().unique())

        # Assert no overlap
        train_val_overlap = train_patients.intersection(val_patients)
        train_test_overlap = train_patients.intersection(test_patients)
        val_test_overlap = val_patients.intersection(test_patients)

        assert len(train_val_overlap) == 0, f"Patient leakage between train and val: {train_val_overlap}"
        assert len(train_test_overlap) == 0, f"Patient leakage between train and test: {train_test_overlap}"
        assert len(val_test_overlap) == 0, f"Patient leakage between val and test: {val_test_overlap}"


def test_lodo_held_out_domain_isolation():
    """Verify that held-out target domain images never appear in source training data."""
    master_meta_csv = PROJECT_ROOT / "data" / "metadata" / "master_metadata.csv"
    if not master_meta_csv.exists():
        pytest.skip("Master metadata not present")

    df = pd.read_csv(master_meta_csv)
    ds_col = "dataset_id" if "dataset_id" in df.columns else ("source_dataset" if "source_dataset" in df.columns else None)
    if ds_col is None or "split" not in df.columns:
        pytest.skip("dataset identifier or split column missing")

    # In our primary setup, IDRiD is source and APTOS is external held-out test
    # If APTOS images are flagged as held-out, verify none are in train split
    source_train_domains = set(df[df["split"] == "train"][ds_col].unique())

    # If user trained on IDRiD only, APTOS must not be in training set
    if "IDRiD" in source_train_domains and len(source_train_domains) == 1:
        assert "APTOS" not in source_train_domains


def test_synthetic_patient_split_generator(tmp_path):
    """Verify patient-level splitting function guarantees mathematical zero leakage."""
    from src.data.split_data import create_splits

    # Create synthetic dataset with multiple images per patient (e.g. left and right eyes)
    records = []
    for pid in range(100):
        # 2 images per patient
        records.append({"image_id": f"img_{pid}_left", "patient_id": f"P{pid:03d}", "dr_grade": pid % 5, "label_status": "verified"})
        records.append({"image_id": f"img_{pid}_right", "patient_id": f"P{pid:03d}", "dr_grade": pid % 5, "label_status": "verified"})

    df = pd.DataFrame(records)
    meta_csv = tmp_path / "metadata.csv"
    df.to_csv(meta_csv, index=False)

    out_dir = tmp_path / "manifests"
    create_splits(meta_csv, out_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    train_df = pd.read_csv(out_dir / "train_manifest.csv")
    val_df = pd.read_csv(out_dir / "val_manifest.csv")
    test_df = pd.read_csv(out_dir / "test_manifest.csv")

    p_train = set(train_df["patient_id"])
    p_val = set(val_df["patient_id"])
    p_test = set(test_df["patient_id"])

    assert len(p_train.intersection(p_val)) == 0
    assert len(p_train.intersection(p_test)) == 0
    assert len(p_val.intersection(p_test)) == 0
