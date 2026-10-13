"""RetinaGuard - Reproducibility utilities."""

import os
import random
import platform
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: Whether to enable deterministic mode in PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_hardware_info() -> dict[str, Any]:
    """Collect hardware and environment information."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(get_device()),
        "cpu_count": os.cpu_count(),
        "ram_gb": None,
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / (1024**3), 2
        )

    # Try to get RAM info
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        pass

    return info


def save_experiment_config(
    config: dict[str, Any],
    experiment_dir: str | Path,
    experiment_id: str | None = None,
) -> Path:
    """Save experiment configuration to a JSON file.

    Args:
        config: Experiment configuration dictionary.
        experiment_dir: Directory to save the config.
        experiment_id: Optional experiment identifier.

    Returns:
        Path to saved config file.
    """
    exp_dir = Path(experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    if experiment_id is None:
        experiment_id = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

    full_config = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "hardware": get_hardware_info(),
        "config": config,
    }

    config_path = exp_dir / f"{experiment_id}_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=2, default=str)

    return config_path


def get_disk_space(path: str | Path = ".") -> dict[str, float]:
    """Get disk space in GB for the given path."""
    import shutil
    total, used, free = shutil.disk_usage(str(path))
    return {
        "total_gb": round(total / (1024**3), 2),
        "used_gb": round(used / (1024**3), 2),
        "free_gb": round(free / (1024**3), 2),
    }
