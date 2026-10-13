"""RetinaGuard - Inference Module.

Run DR grade predictions on individual images or batches.

Usage:
    python -m src.inference.predict --image path/to/image.jpg
    python -m src.inference.predict --image-dir path/to/dir --checkpoint path/to/model.pth
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline import DRClassifier
from src.utils.logging import get_logger
from src.utils.reproducibility import get_device
from src.utils.checkpointing import load_checkpoint

logger = get_logger("predict")

DR_GRADE_DESCRIPTIONS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}


class DRPredictor:
    """Runs inference with a trained DR classifier."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        architecture: str = "resnet18",
        image_size: int = 512,
    ):
        self.device = get_device()
        self.image_size = image_size

        self.model = DRClassifier(
            architecture=architecture, num_classes=5, pretrained=False,
        )

        if checkpoint_path:
            load_checkpoint(checkpoint_path, self.model, device=self.device)
            logger.info(f"Loaded model from: {checkpoint_path}")

        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> dict:
        """Predict DR grade for a single image.

        Args:
            image_path: Path to retinal fundus image.

        Returns:
            Prediction dictionary with grade, confidence, and probabilities.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            return {"error": f"Image not found: {image_path}"}

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": f"Cannot read image: {e}"}

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))

        return {
            "image": str(image_path),
            "predicted_grade": pred_class,
            "grade_description": DR_GRADE_DESCRIPTIONS[pred_class],
            "confidence": float(probs[pred_class]),
            "probabilities": {
                DR_GRADE_DESCRIPTIONS[i]: float(probs[i]) for i in range(5)
            },
        }

    def predict_batch(self, image_paths: list[str | Path]) -> list[dict]:
        """Predict DR grades for multiple images."""
        return [self.predict(p) for p in image_paths]


def main():
    parser = argparse.ArgumentParser(description="RetinaGuard DR Prediction")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image-dir", type=str, help="Directory of images")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--architecture", type=str, default="resnet18")
    parser.add_argument("--output", type=str, help="Output JSON path")
    args = parser.parse_args()

    predictor = DRPredictor(
        checkpoint_path=args.checkpoint,
        architecture=args.architecture,
    )

    results = []

    if args.image:
        result = predictor.predict(args.image)
        results.append(result)
        logger.info(json.dumps(result, indent=2))

    elif args.image_dir:
        img_dir = Path(args.image_dir)
        image_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        logger.info(f"Processing {len(image_paths)} images...")
        results = predictor.predict_batch(image_paths)

    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
