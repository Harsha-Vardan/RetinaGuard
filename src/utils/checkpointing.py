"""RetinaGuard - Checkpoint management utilities."""

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
    checkpoint_dir: str | Path,
    filename: str = "checkpoint.pth",
    scheduler: Any = None,
) -> Path:
    """Save a training checkpoint.

    Args:
        model: PyTorch model.
        optimizer: Optimizer state.
        epoch: Current epoch number.
        metrics: Current evaluation metrics.
        config: Training configuration.
        checkpoint_dir: Directory to save checkpoint.
        filename: Checkpoint filename.
        scheduler: Optional learning rate scheduler.

    Returns:
        Path to saved checkpoint.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    ckpt_path = ckpt_dir / filename
    torch.save(checkpoint, ckpt_path)

    # Save metadata alongside
    meta_path = ckpt_dir / f"{Path(filename).stem}_meta.json"
    meta = {
        "epoch": epoch,
        "metrics": {k: v for k, v in metrics.items()
                    if not isinstance(v, (list, dict)) or k == "per_class"},
        "filename": filename,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    return ckpt_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: PyTorch model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        device: Device to map tensors to.

    Returns:
        Dictionary with checkpoint metadata (epoch, metrics, config).
    """
    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
    }
