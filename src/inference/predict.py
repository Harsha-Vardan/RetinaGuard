"""RetinaGuard - Unified Inference and Clinical Attribution Predictor.

Supports both Baseline DRClassifiers and Flagship RetinaGuardModels.
Generates DR grade predictions, class probabilities, and visual
lesion attention maps.

Usage:
    python -m src.inference.predict --image data/raw/IDRiD/images/IDRiD_001.jpg
    python -m src.inference.predict --image test.jpg --checkpoint experiments/retinaguard/best_retinaguard.pth --save-heatmap heatmap.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline import DRClassifier
from src.models.retinaguard import RetinaGuardModel
from src.explainability.attribution import extract_lesion_attention_map, generate_explanation_figure
from src.utils.logging import get_logger
from src.utils.reproducibility import get_device

logger = get_logger("predict")

DR_GRADE_DESCRIPTIONS = {
    0: "No DR",
    1: "Mild DR (Microaneurysms only)",
    2: "Moderate DR (Microaneurysms + Exudates/Hemorrhages)",
    3: "Severe DR (4-2-1 Rule / Extensive Hemorrhages)",
    4: "Proliferative DR (Neovascularization)",
}


class DRPredictor:
    """Runs inference and visual attribution with a trained DR classifier."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model_type: str = "auto",
        architecture: str = "resnet18",
        image_size: int = 512,
    ):
        self.device = get_device()
        self.image_size = image_size
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path

        # If checkpoint exists, inspect it to detect model type if set to auto
        ckpt_data = None
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if self.model_type == "auto":
                state_dict = ckpt_data.get("model_state_dict", ckpt_data)
                if any("lesion_attention" in k or "disease_classifier" in k for k in state_dict.keys()):
                    self.model_type = "retinaguard"
                else:
                    self.model_type = "baseline"
                logger.info(f"Auto-detected model architecture: {self.model_type}")

        if self.model_type == "retinaguard":
            self.model = RetinaGuardModel(
                architecture=architecture, num_classes=5, num_domains=2, pretrained=False
            )
        else:
            self.model_type = "baseline"
            self.model = DRClassifier(
                architecture=architecture, num_classes=5, pretrained=False
            )

        if ckpt_data is not None:
            state_dict = ckpt_data.get("model_state_dict", ckpt_data)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded {self.model_type} weights from: {checkpoint_path}")

        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(
        self,
        image_path: Union[str, Path],
        save_heatmap_path: Optional[Union[str, Path]] = None,
        ground_truth_mask_path: Optional[Union[str, Path]] = None,
    ) -> dict:
        """Predict DR grade and generate visual explanation for a single image.

        Args:
            image_path: Path to retinal fundus image.
            save_heatmap_path: Optional file path to save multi-panel explanation figure.
            ground_truth_mask_path: Optional ground truth lesion mask for comparison.

        Returns:
            Dictionary with prediction results and attribution metadata.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            return {"error": f"Image not found: {image_path}"}

        try:
            raw_img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": f"Cannot read image: {e}"}

        tensor = self.transform(raw_img).unsqueeze(0).to(self.device)

        # Forward pass
        if self.model_type == "retinaguard":
            out = self.model(tensor)
            logits = out["disease_logits"]
            attn_mask = out.get("attn_mask")
        else:
            logits = self.model(tensor)
            attn_mask = None

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        result = {
            "image": str(image_path),
            "model_type": self.model_type,
            "predicted_grade": pred_class,
            "grade_description": DR_GRADE_DESCRIPTIONS[pred_class],
            "confidence": round(confidence, 4),
            "referable_dr": bool(pred_class >= 2),
            "probabilities": {
                f"Grade {i}": round(float(probs[i]), 4) for i in range(5)
            },
        }

        # Handle heatmap extraction & visualization if requested or available
        gt_mask = None
        if ground_truth_mask_path and Path(ground_truth_mask_path).exists():
            gt_mask = cv2.imread(str(ground_truth_mask_path), cv2.IMREAD_GRAYSCALE)

        if save_heatmap_path:
            img_np = np.array(raw_img)
            if self.model_type == "retinaguard":
                heatmap = extract_lesion_attention_map(self.model, tensor, target_size=(img_np.shape[0], img_np.shape[1]))
            else:
                # Fallback to Grad-CAM for baseline
                from src.explainability.attribution import GradCAM
                grad_cam = GradCAM(self.model, self.model.backbone.layer4)
                heatmap = grad_cam.generate(tensor, target_class=pred_class)

            generate_explanation_figure(
                image_np=img_np,
                heatmap=heatmap,
                ground_truth_mask=gt_mask,
                save_path=save_heatmap_path,
                predicted_grade=pred_class,
                confidence=confidence,
            )
            result["explanation_figure"] = str(save_heatmap_path)

        return result


def main():
    parser = argparse.ArgumentParser(description="RetinaGuard DR Inference & Explanation")
    parser.add_argument("--image", type=str, help="Path to single fundus image")
    parser.add_argument("--image-dir", type=str, help="Directory containing fundus images")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, default="auto", choices=["auto", "baseline", "retinaguard"])
    parser.add_argument("--save-heatmap", type=str, default=None, help="Path or directory to save explanation figures")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path for predictions")
    args = parser.parse_args()

    predictor = DRPredictor(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
    )

    if args.image:
        res = predictor.predict(args.image, save_heatmap_path=args.save_heatmap)
        print(json.dumps(res, indent=2))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

    elif args.image_dir:
        dir_path = Path(args.image_dir)
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif")
        image_files = []
        for ext in extensions:
            image_files.extend(list(dir_path.glob(ext)))

        logger.info(f"Found {len(image_files)} images in {dir_path}")
        results = []
        for img_path in image_files:
            hm_path = None
            if args.save_heatmap:
                out_dir = Path(args.save_heatmap)
                out_dir.mkdir(parents=True, exist_ok=True)
                hm_path = out_dir / f"{img_path.stem}_explanation.png"

            res = predictor.predict(img_path, save_heatmap_path=hm_path)
            results.append(res)

        print(f"Processed {len(results)} images.")
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
