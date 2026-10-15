"""RetinaGuard - Data Splitting.

Creates reproducible train/val/test splits with patient-aware splitting
when patient identifiers are available.

Usage:
    python -m src.data.split_data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger("split_data")


def create_splits(
    metadata_path: str | Path,
    output_dir: str | Path | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_col: str = "dr_grade",
    group_col: str = "patient_id",
) -> pd.DataFrame:
    """Create train/val/test splits.

    Uses patient-level splitting when patient_id is available,
    falls back to stratified splitting otherwise.

    Args:
        metadata_path: Path to master_metadata.csv.
        output_dir: Directory to save split manifests.
        train_ratio: Training set proportion.
        val_ratio: Validation set proportion.
        test_ratio: Test set proportion.
        seed: Random seed.
        stratify_col: Column to stratify by.
        group_col: Column for group-level splitting (e.g. patient_id).

    Returns:
        DataFrame with updated split assignments.
    """
    df = pd.read_csv(metadata_path)
    if "split" not in df.columns:
        df["split"] = None
    df["split"] = df["split"].astype(object)
    logger.info(f"Total records: {len(df)}")

    # Only split records that have labels and aren't already excluded
    mask = df[stratify_col].notna()
    if "label_status" in df.columns:
        mask &= df["label_status"] != "excluded"

    splittable = df[mask].copy()
    unsplittable = df[~mask].copy()

    logger.info(f"Splittable records (with labels): {len(splittable)}")
    logger.info(f"Excluded from splitting: {len(unsplittable)}")

    if len(splittable) == 0:
        logger.warning("No records with labels to split!")
        return df

    np.random.seed(seed)

    # Check for existing splits from source datasets
    if "split" in splittable.columns and splittable["split"].notna().any():
        logger.info("Using existing split assignments from source datasets")
        train_mask = splittable["split"] == "train"
        train_records = splittable[train_mask]
        if len(train_records) > 2 and ("val" not in splittable["split"].values or splittable["split"].eq("val").sum() == 0):
            logger.info(f"Carving validation set out of {len(train_records)} training records")
            val_fraction = val_ratio / (train_ratio + val_ratio)
            train_val_splits = _stratified_split(
                train_records, stratify_col, group_col,
                train_ratio=1.0 - val_fraction, val_ratio=val_fraction, seed=seed
            )
            val_sub_mask = train_val_splits.isin(["val", "test"])
            splittable.loc[train_records[val_sub_mask].index, "split"] = "val"

        # Respect existing splits but handle records without splits
        needs_split = splittable[splittable["split"].isna()]
        if len(needs_split) > 0:
            logger.info(f"Assigning splits to {len(needs_split)} unassigned records")
            assigned = _stratified_split(
                needs_split, stratify_col, group_col,
                train_ratio, val_ratio, seed
            )
            splittable.loc[needs_split.index, "split"] = assigned
    else:
        # Full split
        splits = _stratified_split(
            splittable, stratify_col, group_col,
            train_ratio, val_ratio, seed
        )
        splittable["split"] = splits

    # Combine
    df.loc[splittable.index, "split"] = splittable["split"]
    df.loc[unsplittable.index, "split"] = "excluded"

    # Strict Patient Isolation Postcondition: Enforce that no patient ID spans multiple splits
    if group_col in df.columns:
        valid_groups = df[df["split"].isin(["train", "val", "test"]) & df[group_col].notna()]
        patient_splits = valid_groups.groupby(group_col)["split"].unique()
        leaked = patient_splits[patient_splits.apply(lambda s: len(set(s)) > 1)]
        if len(leaked) > 0:
            logger.warning(f"Enforcing strict patient isolation for {len(leaked)} split-spanning patients...")
            for pid, s_list in leaked.items():
                chosen_split = "train" if "train" in s_list else s_list[0]
                df.loc[df[group_col] == pid, "split"] = chosen_split

    # Log distribution
    split_counts = df["split"].value_counts()
    logger.info(f"\nSplit distribution:")
    for split, count in split_counts.items():
        logger.info(f"  {split}: {count}")

    # Per-split class distribution
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        if len(split_df) > 0 and split_df[stratify_col].notna().any():
            dist = split_df[stratify_col].value_counts().sort_index()
            logger.info(f"  {split} grade distribution: {dict(dist)}")

    # Save
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(metadata_path, index=False)
        logger.info(f"Updated metadata saved to: {metadata_path}")

        # Save per-split manifests
        for split in ["train", "val", "test"]:
            split_df = df[df["split"] == split]
            if len(split_df) > 0:
                manifest_path = output_dir / f"{split}_manifest.csv"
                split_df.to_csv(manifest_path, index=False)
                logger.info(f"Saved {split} manifest: {manifest_path} ({len(split_df)} records)")

    return df


def _stratified_split(
    df: pd.DataFrame, stratify_col: str, group_col: str,
    train_ratio: float, val_ratio: float, seed: int,
) -> pd.Series:
    """Perform stratified splitting, with optional group awareness."""
    labels = df[stratify_col].astype(int).values
    n = len(df)

    has_groups = (
        group_col in df.columns
        and df[group_col].notna().any()
        and df[group_col].nunique() > 1
    )

    splits = pd.Series("train", index=df.index)
    test_ratio = max(0.0, round(1.0 - train_ratio - val_ratio, 6))

    if has_groups:
        fallback = pd.Series([f"unknown_{i}" for i in range(len(df))], index=df.index)
        groups = df[group_col].fillna(fallback).astype(str).values
        n_unique_groups = len(np.unique(groups))
        logger.info(f"Using patient-aware splitting ({n_unique_groups} groups)")
        try:
            if test_ratio < 0.005:
                # 2-way split directly between train and val
                val_frac = max(0.01, min(0.99, val_ratio))
                n_val_splits = max(2, int(round(1.0 / val_frac)))
                n_val_splits = min(n_val_splits, n_unique_groups)
                if n_val_splits >= 2:
                    sgkf = StratifiedGroupKFold(n_splits=n_val_splits, shuffle=True, random_state=seed)
                    for train_idx, val_idx in sgkf.split(np.zeros(n), labels, groups):
                        splits.iloc[val_idx] = "val"
                        break
                else:
                    has_groups = False
            else:
                # 3-way split: first separate test, then split remaining into train/val
                n_test_splits = max(2, int(round(1.0 / test_ratio)))
                n_test_splits = min(n_test_splits, n_unique_groups)
                if n_test_splits >= 2:
                    sgkf = StratifiedGroupKFold(n_splits=n_test_splits, shuffle=True, random_state=seed)
                    for train_idx, test_idx in sgkf.split(np.zeros(n), labels, groups):
                        splits.iloc[test_idx] = "test"
                        remaining_labels = labels[train_idx]
                        remaining_groups = groups[train_idx]
                        val_frac = val_ratio / (train_ratio + val_ratio)
                        n_val_splits = max(2, int(round(1.0 / max(0.01, val_frac))))
                        rem_groups_count = len(np.unique(remaining_groups))
                        n_val_splits = min(n_val_splits, rem_groups_count)
                        if n_val_splits >= 2:
                            sgkf_val = StratifiedGroupKFold(n_splits=n_val_splits, shuffle=True, random_state=seed)
                            for tr_idx, val_idx in sgkf_val.split(np.zeros(len(train_idx)), remaining_labels, remaining_groups):
                                splits.iloc[train_idx[val_idx]] = "val"
                                break
                        break
                else:
                    has_groups = False
        except Exception as e:
            logger.warning(f"Patient-aware splitting failed: {e}. Falling back to stratified.")
            has_groups = False

    if not has_groups:
        try:
            if test_ratio < 0.005:
                val_frac = max(0.01, min(0.99, val_ratio))
                n_splits = max(2, int(round(1.0 / val_frac)))
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                for train_idx, val_idx in skf.split(np.zeros(n), labels):
                    splits.iloc[val_idx] = "val"
                    break
            else:
                skf = StratifiedKFold(n_splits=max(2, int(round(1.0 / (1.0 - train_ratio)))), shuffle=True, random_state=seed)
                for train_idx, test_val_idx in skf.split(np.zeros(n), labels):
                    splits.iloc[test_val_idx[:len(test_val_idx)//2]] = "val"
                    splits.iloc[test_val_idx[len(test_val_idx)//2:]] = "test"
                    break
        except Exception:
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            splits.iloc[n_train:n_train+n_val] = "val"
            if test_ratio >= 0.005:
                splits.iloc[n_train+n_val:] = "test"

    return splits


def main():
    parser = argparse.ArgumentParser(description="RetinaGuard Data Splitter")
    parser.add_argument("--metadata", type=str,
                       default="data/metadata/master_metadata.csv")
    parser.add_argument("--output-dir", type=str, default="data/manifests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata_path = PROJECT_ROOT / args.metadata
    output_dir = PROJECT_ROOT / args.output_dir

    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        sys.exit(1)

    create_splits(metadata_path, output_dir, seed=args.seed)


if __name__ == "__main__":
    main()

