"""RetinaGuard - Synthetic / Mock Retinal Dataset Generator.

Generates realistic synthetic retinal fundus images, lesion masks, and
ground-truth labels formatted identically to official IDRiD and APTOS datasets.
Used for pipeline verification, CPU smoke testing, and unit tests.

Usage:
    python -m src.data.create_mock_data --num-idrid 20 --num-aptos 20
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger("create_mock_data")


def generate_synthetic_fundus(
    height: int = 512,
    width: int = 512,
    dr_grade: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic retinal fundus image and lesion mask.

    Args:
        height: Image height.
        width: Image width.
        dr_grade: DR severity grade (0 to 4).
        seed: Random seed for reproducibility.

    Returns:
        tuple of (BGR image, uint8 binary lesion mask).
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Canvas with black background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    # 2. Retinal Field of View (FOV)
    center = (width // 2, height // 2)
    radius = int(min(height, width) * 0.44)

    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    fov_mask = dist_from_center <= radius

    # Base retinal gradient (reddish-orange with natural shading)
    base_r = np.clip(180 - (dist_from_center / radius * 50) + np.random.normal(0, 5, (height, width)), 0, 255).astype(np.uint8)
    base_g = np.clip(70 - (dist_from_center / radius * 25) + np.random.normal(0, 3, (height, width)), 0, 255).astype(np.uint8)
    base_b = np.clip(30 - (dist_from_center / radius * 15) + np.random.normal(0, 2, (height, width)), 0, 255).astype(np.uint8)

    img[..., 0] = np.where(fov_mask, base_b, 0)
    img[..., 1] = np.where(fov_mask, base_g, 0)
    img[..., 2] = np.where(fov_mask, base_r, 0)

    # 3. Optic Disc (yellowish oval on nasal side)
    od_x = center[0] + int(radius * 0.48)
    od_y = center[1] + int(np.random.uniform(-0.1, 0.1) * radius)
    cv2.ellipse(img, (od_x, od_y), (int(radius * 0.14), int(radius * 0.17)), 0, 0, 360, (140, 220, 250), -1)

    # 4. Fovea / Macula (darker reddish spot temporal to disc)
    fovea_x = center[0] - int(radius * 0.25)
    fovea_y = center[1]
    cv2.circle(img, (fovea_x, fovea_y), int(radius * 0.12), (20, 45, 130), -1)

    # 5. Blood Vessels radiating from optic disc
    for angle in np.linspace(0.8 * np.pi, 1.4 * np.pi, 6):
        pts = [(od_x, od_y)]
        curr_x, curr_y = od_x, od_y
        for step in range(5):
            curr_x += int(np.cos(angle) * (radius * 0.15) + np.random.randint(-10, 10))
            curr_y += int(np.sin(angle) * (radius * 0.15) + np.random.randint(-10, 10))
            pts.append((curr_x, curr_y))
        for i in range(len(pts) - 1):
            py = min(height - 1, max(0, pts[i][1]))
            px = min(width - 1, max(0, pts[i][0]))
            if fov_mask[py, px]:
                cv2.line(img, pts[i], pts[i + 1], (15, 25, 90), thickness=max(1, 4 - i))

    # 6. Add DR lesions based on DR Grade
    if dr_grade >= 1:
        # Microaneurysms: small dark red dots
        num_ma = dr_grade * 8
        for _ in range(num_ma):
            lx = int(center[0] + np.random.uniform(-0.6, 0.6) * radius)
            ly = int(center[1] + np.random.uniform(-0.6, 0.6) * radius)
            if 0 <= lx < width and 0 <= ly < height and fov_mask[ly, lx]:
                cv2.circle(img, (lx, ly), np.random.randint(2, 4), (10, 15, 70), -1)
                cv2.circle(mask, (lx, ly), np.random.randint(2, 4), 255, -1)

    if dr_grade >= 2:
        # Hard Exudates: bright yellowish deposits
        num_he = (dr_grade - 1) * 10
        for _ in range(num_he):
            lx = int(center[0] + np.random.uniform(-0.5, 0.5) * radius)
            ly = int(center[1] + np.random.uniform(-0.5, 0.5) * radius)
            if 0 <= lx < width and 0 <= ly < height and fov_mask[ly, lx]:
                cv2.circle(img, (lx, ly), np.random.randint(2, 5), (120, 240, 255), -1)
                cv2.circle(mask, (lx, ly), np.random.randint(2, 5), 255, -1)

    if dr_grade >= 3:
        # Hemorrhages: larger irregular blotches
        num_hem = (dr_grade - 2) * 6
        for _ in range(num_hem):
            lx = int(center[0] + np.random.uniform(-0.5, 0.5) * radius)
            ly = int(center[1] + np.random.uniform(-0.5, 0.5) * radius)
            if 0 <= lx < width and 0 <= ly < height and fov_mask[ly, lx]:
                cv2.ellipse(img, (lx, ly), (np.random.randint(5, 12), np.random.randint(4, 9)),
                            np.random.randint(0, 180), 0, 360, (10, 10, 50), -1)
                cv2.ellipse(mask, (lx, ly), (np.random.randint(5, 12), np.random.randint(4, 9)),
                            np.random.randint(0, 180), 0, 360, 255, -1)

    if dr_grade >= 4:
        # Neovascularization / extensive proliferative changes
        for _ in range(5):
            lx = int(od_x + np.random.uniform(-0.2, 0.2) * radius)
            ly = int(od_y + np.random.uniform(-0.2, 0.2) * radius)
            if 0 <= lx < width and 0 <= ly < height and fov_mask[ly, lx]:
                cv2.circle(img, (lx, ly), np.random.randint(8, 16), (20, 30, 80), -1)
                cv2.circle(mask, (lx, ly), np.random.randint(8, 16), 255, -1)

    # Slight Gaussian blur to mimic retinal camera optics
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return img, mask


def create_mock_idrid(raw_dir: Path, num_images: int = 25) -> Path:
    """Create a mock IDRiD dataset directory structure.

    Args:
        raw_dir: Root raw data directory.
        num_images: Total number of images to generate.

    Returns:
        Path to IDRiD directory.
    """
    idrid_dir = raw_dir / "IDRiD"
    train_img_dir = idrid_dir / "A. Disease Grading" / "1. Original Images" / "a. Training Set"
    test_img_dir = idrid_dir / "A. Disease Grading" / "1. Original Images" / "b. Testing Set"
    gt_dir = idrid_dir / "A. Disease Grading" / "2. Groundtruths"
    mask_dir = idrid_dir / "C. Localization and Segmentation" / "1. Microaneurysms"

    train_img_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    num_train = int(num_images * 0.7)
    num_test = num_images - num_train

    train_records = []
    test_records = []

    # Generate training set
    for i in range(1, num_train + 1):
        img_id = f"IDRiD_{i:03d}"
        grade = (i - 1) % 5  # Balances grades 0, 1, 2, 3, 4
        img, mask = generate_synthetic_fundus(height=512, width=512, dr_grade=grade, seed=i)

        img_path = train_img_dir / f"{img_id}.jpg"
        cv2.imwrite(str(img_path), img)

        if grade > 0:
            mask_path = mask_dir / f"{img_id}_MA.tif"
            cv2.imwrite(str(mask_path), mask)

        train_records.append({
            "Image name": img_id,
            "Retinopathy grade": grade,
            "Risk of macular edema": 1 if grade >= 2 else 0,
        })

    # Generate testing set
    for i in range(num_train + 1, num_images + 1):
        img_id = f"IDRiD_{i:03d}"
        grade = (i - 1) % 5
        img, mask = generate_synthetic_fundus(height=512, width=512, dr_grade=grade, seed=i + 100)

        img_path = test_img_dir / f"{img_id}.jpg"
        cv2.imwrite(str(img_path), img)

        test_records.append({
            "Image name": img_id,
            "Retinopathy grade": grade,
            "Risk of macular edema": 1 if grade >= 2 else 0,
        })

    # Save Groundtruth CSVs
    pd.DataFrame(train_records).to_csv(
        gt_dir / "a. IDRiD_Disease Grading_Training Labels.csv", index=False
    )
    pd.DataFrame(test_records).to_csv(
        gt_dir / "b. IDRiD_Disease Grading_Testing Labels.csv", index=False
    )

    logger.info(f"Created mock IDRiD dataset: {num_train} train, {num_test} test images in {idrid_dir}")
    return idrid_dir


def create_mock_aptos(raw_dir: Path, num_images: int = 25) -> Path:
    """Create a mock APTOS dataset directory structure.

    Args:
        raw_dir: Root raw data directory.
        num_images: Total number of images to generate.

    Returns:
        Path to APTOS directory.
    """
    aptos_dir = raw_dir / "APTOS"
    train_dir = aptos_dir / "train_images"
    test_dir = aptos_dir / "test_images"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    num_train = int(num_images * 0.7)
    num_test = num_images - num_train

    train_records = []
    test_records = []

    for i in range(1, num_train + 1):
        id_code = f"aptos_{i:04x}"
        grade = (i - 1) % 5
        img, _ = generate_synthetic_fundus(height=512, width=512, dr_grade=grade, seed=i + 500)

        cv2.imwrite(str(train_dir / f"{id_code}.png"), img)
        train_records.append({"id_code": id_code, "diagnosis": grade})

    for i in range(num_train + 1, num_images + 1):
        id_code = f"aptos_{i:04x}"
        img, _ = generate_synthetic_fundus(height=512, width=512, dr_grade=0, seed=i + 800)
        cv2.imwrite(str(test_dir / f"{id_code}.png"), img)
        test_records.append({"id_code": id_code})

    pd.DataFrame(train_records).to_csv(aptos_dir / "train.csv", index=False)
    pd.DataFrame(test_records).to_csv(aptos_dir / "test.csv", index=False)

    logger.info(f"Created mock APTOS dataset: {num_train} train, {num_test} test images in {aptos_dir}")
    return aptos_dir


def main():
    parser = argparse.ArgumentParser(description="RetinaGuard Mock Dataset Generator")
    parser.add_argument("--num-idrid", type=int, default=25, help="Number of IDRiD mock images")
    parser.add_argument("--num-aptos", type=int, default=25, help="Number of APTOS mock images")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Raw data directory")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / args.output_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    create_mock_idrid(raw_dir, args.num_idrid)
    create_mock_aptos(raw_dir, args.num_aptos)
    logger.info("Mock dataset generation completed successfully.")


if __name__ == "__main__":
    main()
