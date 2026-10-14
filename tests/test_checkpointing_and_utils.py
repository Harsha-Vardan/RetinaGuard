"""RetinaGuard - Tests for Checkpointing, Reproducibility, and Utilities."""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.utils.checkpointing import load_checkpoint, save_checkpoint
from src.utils.logging import get_logger
from src.utils.reproducibility import get_device, set_seed


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def test_set_seed_deterministic():
    """Verify that set_seed ensures reproducible random tensor generation."""
    set_seed(42)
    t1 = torch.randn(5, 5)

    set_seed(42)
    t2 = torch.randn(5, 5)

    assert torch.allclose(t1, t2)


def test_get_device_returns_torch_device():
    """Verify that get_device returns a valid torch.device object."""
    dev = get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda")


def test_get_logger_creation():
    """Verify logger creation and handler configuration."""
    logger = get_logger("test_module")
    assert logger.name == "test_module"
    assert len(logger.handlers) > 0


def test_save_and_load_checkpoint(tmp_path):
    """Verify atomic checkpoint save and load cycle."""
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    ckpt_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=5,
        metrics={"qwk": 0.88},
        config={"architecture": "dummy", "lr": 0.01},
        checkpoint_dir=tmp_path,
        filename="test_checkpoint.pth",
    )

    assert ckpt_path.exists()

    # Load into new model instance
    new_model = DummyModel()
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)

    loaded_ckpt = load_checkpoint(ckpt_path, model=new_model, optimizer=new_optimizer)

    assert loaded_ckpt["epoch"] == 5
    assert loaded_ckpt["metrics"]["qwk"] == 0.88

    # Weights should match exactly
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)
